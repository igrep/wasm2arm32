use indexmap::IndexMap;
use std::{
    any::Any,
    collections::HashMap,
    path::{Path, PathBuf},
    fmt,
    env, fs,
    ptr::NonNull,
    sync::{Arc, RwLock},
    io::{Write, stdout},
    io,
    process,
};
use wasmer_runtime_core::{
    backend::{
        sys::Memory, Architecture, CacheGen, CompilerConfig, ExceptionTable, Features,
        InlineBreakpoint, MemoryBoundCheckMode, RunnableModule, Token,
    },
    cache::{Artifact, Error as CacheError},
    codegen::{
        BreakpointMap, DebugMetadata, Event, FunctionCodeGenerator, MiddlewareChain,
        ModuleCodeGenerator, WasmSpan,
    },
    fault,
    module::{ModuleInfo, ModuleInner, StringTable},
    parse,
    state::ModuleStateMap,
    structures::Map,
    typed_func::Wasm,
    types::{FuncIndex, FuncSig, LocalFuncIndex, SigIndex},
    vm,
};
use wasmparser::{Operator, Type as WpType};

fn main() {
    let argument = env::args().skip(1).next().expect("No wasm file specified!");
    let wasm_data = fs::read(&argument).expect(&format!("Error reading {:?}", &argument));
    compile(&argument, &wasm_data);
}

#[test]
fn test_example() {
    let path = "tmp/data/example.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    compile(path, &wasm_data);
}

#[test]
fn test_i32add() {
    let path = "tmp/data/i32add.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    let obj_path = compile(path, &wasm_data);
    let status = process::Command::new(&obj_path).status().expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(46));
}

#[test]
fn test_i32const() {
    let path = "tmp/data/i32const.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    let obj_path = compile(path, &wasm_data);
    let status = process::Command::new(&obj_path).status().expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(42));
}

#[test]
fn test_empty() {
    let path = "tmp/data/empty.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    compile(path, &wasm_data);
}

pub fn compile<P: AsRef<Path> + fmt::Display>(path: P, wasm_data: &[u8]) -> PathBuf {
    let mut mcg = Arm32ModuleCodeGenerator::new();
    let mut middlewares = MiddlewareChain::new();
    let compiler_config = CompilerConfig {
        /// Symbol information generated from emscripten; used for more detailed debug messages
        symbol_map: None,

        /// How to make the decision whether to emit bounds checks for memory accesses.
        memory_bound_check_mode: MemoryBoundCheckMode::Default,

        /// Whether to generate explicit native stack checks against `stack_lower_bound` in `InternalCtx`.
        ///
        /// Usually it's adequate to use hardware memory protection mechanisms such as `mprotect` on Unix to
        /// prevent stack overflow. But for low-level environments, e.g. the kernel, faults are generally
        /// not expected and relying on hardware memory protection would add too much complexity.
        enforce_stack_check: false,

        /// Whether to enable state tracking. Necessary for managed mode.
        track_state: false,

        /// Whether to enable full preemption checkpoint generation.
        ///
        /// This inserts checkpoints at critical locations such as loop backedges and function calls,
        /// allowing preemptive unwinding/task switching.
        ///
        /// When enabled there can be a small amount of runtime performance overhead.
        full_preemption: false,

        /// Always choose a unique bit representation for NaN.
        /// Enabling this makes execution deterministic but increases runtime overhead.
        nan_canonicalization: false,

        /// Turns on verification that is done by default when `debug_assertions` are enabled
        /// (for example in 'debug' builds). Disabling this flag will make compilation faster
        /// in debug mode at the cost of not detecting bugs in the compiler.
        ///
        /// These verifications are disabled by default in 'release' builds.
        enable_verification: false,

        features: Features {
            simd: false,
            threads: false,
        },

        // Target info. Presently only supported by LLVM.
        triple: None,
        cpu_name: None,
        cpu_features: None,

        backend_specific_config: None,

        generate_debug_info: false,
    };
    // Try reading the sample WASM module
    parse::read_module(wasm_data, &mut mcg, &mut middlewares, &compiler_config).expect("Failed to read_module");
    let asm_path = path.as_ref().with_extension("s");
    {
        let asm_out =
            fs::File::create(&asm_path).expect(&format!("Failed to create file {:?}!", &asm_path));
        mcg.asm_writer = Box::new(asm_out);
        mcg.save_asm().unwrap();
    }

    let obj_path = path.as_ref().with_extension("o");
    let status = process::Command::new("gcc")
        .args(&["-o", obj_path.to_string_lossy().as_ref(), asm_path.to_string_lossy().as_ref()])
        .status()
        .expect("Unable to run gcc command!");
    if !status.success() {
        match status.code() {
            Some(code) => panic!("gcc exited with status code: {}", code),
            None       => panic!("gcc terminated by signal")
        }
    }
    return obj_path;
}

static BACKEND_ID: &str = "singlepass";

#[derive(Debug)]
pub struct CodegenError {
    pub message: String,
}

pub struct Arm32ModuleCodeGenerator {
    functions: Vec<Arm32FunctionCode>,
    asm_writer: Box<dyn Write>,
}

impl Arm32ModuleCodeGenerator {
    fn save_asm(&mut self) -> io::Result<()> {
        for fx in &self.functions {
            writeln!(self.asm_writer, ".global {}", fx.name)?
        }
        for fx in &self.functions {
            writeln!(self.asm_writer, "{}", fx.asm)?
        }
        Ok(())
    }
}

impl ModuleCodeGenerator<Arm32FunctionCode, Arm32ExecutionContext, CodegenError>
for Arm32ModuleCodeGenerator
{
    /// Creates a new module code generator.
    fn new() -> Self {
        Arm32ModuleCodeGenerator {
            functions: Vec::new(),
            asm_writer: Box::new(stdout()),
        }
    }

    /// Creates a new module code generator for specified target.
    fn new_with_target(
        _triple: Option<String>,
        _cpu_name: Option<String>,
        _cpu_features: Option<String>,
    ) -> Self {
        unimplemented!("cross compilation is not available for singlepass backend")
    }

    /// Returns the backend id associated with this MCG.
    fn backend_id() -> &'static str {
        BACKEND_ID
    }

    /// It sets if the current compiler requires validation before compilation
    fn requires_pre_validation() -> bool {
        false
    }

    /// Feeds the compiler config.
    fn feed_compiler_config(&mut self, _config: &CompilerConfig) -> Result<(), CodegenError> {
        println!("feed_compiler_config");
        Ok(())
    }

    /// Adds an import function.
    fn feed_import_function(&mut self, sigindex: SigIndex) -> Result<(), CodegenError> {
        println!("feed_import_function: sigindex: {:?}", sigindex);
        Ok(())
    }

    /// Sets the signatures.
    fn feed_signatures(&mut self, signatures: Map<SigIndex, FuncSig>) -> Result<(), CodegenError> {
        println!("feed_import_function: signatures: {:#?}", signatures);
        Ok(())
    }

    /// Sets function signatures.
    fn feed_function_signatures(
        &mut self,
        assoc: Map<FuncIndex, SigIndex>,
    ) -> Result<(), CodegenError> {
        println!("feed_function_signatures: signatures: {:#?}", assoc);
        Ok(())
    }

    /// Checks the precondition for a module.
    fn check_precondition(&mut self, _module_info: &ModuleInfo) -> Result<(), CodegenError> {
        println!("check_precondition");
        Ok(())
    }

    /// Creates a new function and returns the function-scope code generator for it.
    fn next_function(
        &mut self,
        module_info: Arc<RwLock<ModuleInfo>>,
        loc: WasmSpan,
    ) -> Result<&mut Arm32FunctionCode, CodegenError> {
        println!("next_function: loc: {:?}, module_info: {:#?}", loc, module_info);
        let code = Arm32FunctionCode {
            id: self.functions.len(),
            name: "main".into(),
            asm: String::new(),
        };
        println!("next_function: return: {:?}", code);
        self.functions.push(code);
        Ok(self
            .functions
            .last_mut()
            .expect("Assertion failure: can't get mutable reference of code"))
    }

    /// Finalizes this module.
    fn finalize(
        self,
        module_info: &ModuleInfo,
    ) -> Result<
        (
            Arm32ExecutionContext,
            Option<DebugMetadata>,
            Box<dyn CacheGen>,
        ),
        CodegenError,
        > {
            println!("finalize: {:#?}", module_info);
            let cache = DummyCacheGen::DummyCacheGen;
            Ok((Arm32ExecutionContext { hoge: 0 }, None, Box::new(cache)))
    }

    /// Creates a module from cache.
    unsafe fn from_cache(_cache: Artifact, _: Token) -> Result<ModuleInner, CacheError> {
        println!("from_cache");
        Ok(ModuleInner {
            runnable_module: Arc::new(Box::new(Arm32ExecutionContext { hoge: 0 })),
            cache_gen: Box::new(DummyCacheGen::DummyCacheGen),
            info: ModuleInfo {
                memories: Map::new(),
                globals: Map::new(),
                /// Map of table index to table descriptors.
                tables: Map::new(),

                /// Map of imported function index to import name.
                // These are strictly imported and the typesystem ensures that.
                imported_functions: Map::new(),
                /// Map of imported memory index to import name and memory descriptor.
                imported_memories: Map::new(),
                /// Map of imported table index to import name and table descriptor.
                imported_tables: Map::new(),
                /// Map of imported global index to import name and global descriptor.
                imported_globals: Map::new(),

                /// Map of string to export index.
                // Implementation note: this should maintain the order that the exports appear in the
                // Wasm module.  Be careful not to use APIs that may break the order!
                // Side note, because this is public we can't actually guarantee that it will remain
                // in order.
                exports: IndexMap::new(),

                /// Vector of data initializers.
                data_initializers: Vec::new(),
                /// Vector of table initializers.
                elem_initializers: Vec::new(),

                /// Index of optional start function.
                start_func: None,

                /// Map function index to signature index.
                func_assoc: Map::new(),
                /// Map signature index to function signature.
                signatures: Map::new(),
                /// Backend.
                backend: String::new(),

                /// Table of namespace indexes.
                namespace_table: StringTable::new(),
                /// Table of name indexes.
                name_table: StringTable::new(),

                /// Symbol information from emscripten.
                em_symbol_map: None,

                /// Custom sections.
                custom_sections: HashMap::new(),

                /// Flag controlling whether or not debug information for use in a debugger
                /// will be generated.
                generate_debug_info: false,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct Arm32FunctionCode {
    id: usize,
    name: String,
    asm: String,
}

impl FunctionCodeGenerator<CodegenError> for Arm32FunctionCode {
    /// Sets the return type.
    fn feed_return(&mut self, ty: WpType) -> Result<(), CodegenError> {
        println!("feed_return: {:#?}", ty);
        Ok(())
    }

    /// Adds a parameter to the function.
    fn feed_param(&mut self, ty: WpType) -> Result<(), CodegenError> {
        println!("feed_param: {:#?}", ty);
        Ok(())
    }

    /// Adds `n` locals to the function.
    fn feed_local(&mut self, ty: WpType, n: usize, loc: u32) -> Result<(), CodegenError> {
        println!("feed_local: {:#?}, n: {}, loc: {}", ty, n, loc);
        Ok(())
    }

    /// Called before the first call to `feed_opcode`.
    fn begin_body(&mut self, module_info: &ModuleInfo) -> Result<(), CodegenError> {
        println!("begin_body: {:#?}", module_info);
        self.asm.push_str(format!("{}:\n", self.name).as_str());
        Ok(())
    }

    /// Called for each operator.
    fn feed_event(
        &mut self,
        op: Event,
        _module_info: &ModuleInfo,
        source_loc: u32,
    ) -> Result<(), CodegenError> {
        match op {
            Event::Wasm(Operator::I32Const { value }) => {
                self.asm.push_str(format!("  MOV R0, #{}\n", value).as_str());
                self.asm.push_str("  PUSH {R0}\n")
            }
            Event::Wasm(Operator::I32Add) => {
                self.asm.push_str("  POP {R1-R2}\n");
                self.asm.push_str("  ADD R0, R1, R2\n")
            },
            _ =>
                println!("Unknown event: {:#?}, at {:#?}", op, source_loc)
        }
        Ok(())
    }

    /// Finalizes the function.
    fn finalize(&mut self) -> Result<(), CodegenError> {
        println!("finalize");
        self.asm.push_str("  BX lr");
        Ok(())
    }
}

pub struct Arm32ExecutionContext {
    hoge: usize,
}

impl RunnableModule for Arm32ExecutionContext {
    /// This returns a pointer to the function designated by the `local_func_index`
    /// parameter.
    fn get_func(
        &self,
        _info: &ModuleInfo,
        local_func_index: LocalFuncIndex,
    ) -> Option<NonNull<vm::Func>> {
        println!("get_trampoline: {:#?}", local_func_index);
        None
    }

    fn get_module_state_map(&self) -> Option<ModuleStateMap> {
        None
    }

    fn get_breakpoints(&self) -> Option<BreakpointMap> {
        None
    }

    fn get_exception_table(&self) -> Option<&ExceptionTable> {
        None
    }

    unsafe fn patch_local_function(&self, _idx: usize, _target_address: usize) -> bool {
        false
    }

    /// A wasm trampoline contains the necessary data to dynamically call an exported wasm function.
    /// Given a particular signature index, we return a trampoline that is matched with that
    /// signature and an invoke function that can call the trampoline.
    fn get_trampoline(&self, _info: &ModuleInfo, sig_index: SigIndex) -> Option<Wasm> {
        println!("get_trampoline: {:#?}", sig_index);
        None
    }

    /// Trap an error.
    unsafe fn do_early_trap(&self, data: Box<dyn Any + Send>) -> ! {
        fault::begin_unsafe_unwind(data)
    }

    /// Returns the machine code associated with this module.
    fn get_code(&self) -> Option<&[u8]> {
        None
    }

    /// Returns the beginning offsets of all functions, including import trampolines.
    fn get_offsets(&self) -> Option<Vec<usize>> {
        None
    }

    /// Returns the beginning offsets of all local functions.
    fn get_local_function_offsets(&self) -> Option<Vec<usize>> {
        None
    }

    /// Returns the inline breakpoint size corresponding to an Architecture (None in case is not implemented)
    fn get_inline_breakpoint_size(&self, _arch: Architecture) -> Option<usize> {
        None
    }

    /// Attempts to read an inline breakpoint from the code.
    ///
    /// Inline breakpoints are detected by special instruction sequences that never
    /// appear in valid code.
    fn read_inline_breakpoint(
        &self,
        _arch: Architecture,
        _code: &[u8],
    ) -> Option<InlineBreakpoint> {
        None
    }
}

#[derive(Debug)]
pub enum DummyCacheGen {
    DummyCacheGen,
}

unsafe impl Send for DummyCacheGen {}
unsafe impl Sync for DummyCacheGen {}

impl CacheGen for DummyCacheGen {
    fn generate_cache(&self) -> Result<(Box<[u8]>, Memory), CacheError> {
        let empty: [u8; 0] = [];
        let mem = Memory::with_size(1).unwrap();
        Ok((Box::new(empty), mem))
    }
}
