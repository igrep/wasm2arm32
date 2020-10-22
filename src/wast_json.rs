use std::{
    fmt::Display,
    fs,
    fs::File,
    io,
    io::Write,
    path::{Path, PathBuf},
    process,
};

extern crate serde_json;
use serde_derive::Deserialize;

use crate::{asm, core::compile};

#[derive(Deserialize)]
struct WastJson {
    commands: Vec<WastCommand>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WastCommand {
    Module {
        filename: PathBuf,
    },
    AssertReturn {
        line: u32,
        action: WastAction,
        expected: Vec<WastValue>,
    },
    AssertTrap {
        line: u32,
        action: WastAction,
        text: String,
        expected: Vec<WastValue>,
    },
    AssertInvalid {
        line: u32,
        filename: PathBuf,
        text: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WastAction {
    Invoke { field: String, args: Vec<WastValue> },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WastValue {
    I32 { value: Option<String> },
}

impl WastValue {
    fn to_literal(&self) -> Option<&String> {
        match self {
            WastValue::I32 { value } => value.as_ref(),
        }
    }
    fn to_movs_to(&self, register_num: usize) -> Option<String> {
        match self {
            WastValue::I32 { value } => {
                let value = value.as_ref()?.parse::<u32>().ok()?;
                Some(asm::to_movs_to(register_num, value))
            }
        }
    }
}

const RUNTIME_ASM: &str = "\
.data
.balign 4

smsg: .asciz \"Test case at %d SUCCESS.\\n\"
emsg: .asciz \"Test case at %d FAILED. Expected %d, Actual: %d\\n\"

.text

.global fprintf
.global stderr
.global exit

@ assert_equal(actual: usize, expected: usize, line: usize)
assert_equal:
  PUSH {LR}
  CMP R1, R2
  BNE assert_equal_failure

  MOV R2, R0
  LDR R1, addr_smsg
  LDR R0, addr_stderr
  LDR R0, [R0]
  BL fprintf

  POP {LR}
  BX LR

assert_equal_failure:
  PUSH {R2}
  MOV R3, R1
  MOV R2, R0
  LDR R1, addr_emsg
  LDR R0, addr_stderr
  LDR R0, [R0]
  BL fprintf

  MOV R0, #1
  BL exit

addr_smsg: .word smsg
addr_emsg: .word emsg
addr_stderr: .word stderr

.global main
.func main

main:
  PUSH {LR}
";

const RUNTIME_ASM_BOTTOM_LINES: &str = "\
  MOV R0, #0
  POP {LR}
  BX LR
";

fn run_tests_in<P: AsRef<Path> + Display>(json_path: P) {
    let wast_json_file = File::open(&json_path)
        .unwrap_or_else(|e| panic!("Failed to OPEN a WAST JSON file: {:?}", e));
    let wast_json: WastJson = serde_json::from_reader(wast_json_file)
        .unwrap_or_else(|e| panic!("Failed to PARSE a WAST JSON file:\n{:?}", e));

    let mut m_current_out: Option<(PathBuf, File)> = None;
    for command in wast_json.commands {
        match command {
            WastCommand::Module { filename } => {
                if let Some((previous_asm_path, previous_out)) = m_current_out.as_mut() {
                    finalize_tests_asm(previous_out).expect("Failed to finalize tests assembly");
                    drop(previous_out);
                    compile_and_run(previous_asm_path);
                }

                let wasm_path = json_path.as_ref().with_file_name(filename);
                let current_wasm_data =
                    fs::read(&wasm_path).expect(&format!("Error reading {:?}", &wasm_path));

                let asm_path = wasm_path.with_extension("s");
                let current_out = fs::File::create(&asm_path)
                    .expect(&format!("Failed to create file {:?}!", &asm_path));
                write!(&current_out, "{}", compile(&current_wasm_data))
                    .expect("Failed to write compiled wasm program!");
                write!(&current_out, "{}", RUNTIME_ASM).expect("Failed to write RUNTIME_ASM!");
                m_current_out = Some((asm_path, current_out));
            }
            WastCommand::AssertReturn {
                action,
                expected,
                line,
            } => {
                let (_, current_out) = m_current_out
                    .as_mut()
                    .expect("The input WAST JSON doesn't contain a 'module' command before 'assert_return'!");
                write!(
                    current_out,
                    "{}",
                    compile_assert_return(action, expected, line)
                )
                .expect("Failed to write assert_return!");
            }
            WastCommand::AssertTrap {
                line: _,
                action: _,
                text: _,
                expected: _,
            } => { /* do nothing so far */ }
            WastCommand::AssertInvalid {
                line: _,
                filename: _,
                text: _,
            } => { /* do nothing so far */ }
        }
    }
    if let Some((previous_asm_path, previous_out)) = m_current_out.as_mut() {
        finalize_tests_asm(previous_out).expect("Failed to finalize tests assembly");
        drop(previous_out);
        compile_and_run(previous_asm_path);
    }
}

fn compile_and_run<P: AsRef<Path>>(asm_path: P) {
    let obj_path = asm::compile_file(asm_path);
    let status = process::Command::new(&obj_path)
        .status()
        .expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(0));
}

fn finalize_tests_asm(out: &mut File) -> io::Result<()> {
    writeln!(out, "{}", RUNTIME_ASM_BOTTOM_LINES)
}

fn compile_assert_return(action: WastAction, expected: Vec<WastValue>, line: u32) -> String {
    let mut result = "".to_string();
    match action {
        WastAction::Invoke { field, args } => {
            for (i, arg) in args.iter().enumerate() {
                // TODO: Support more than 4 arguments.
                result.push_str(
                    &arg.to_movs_to(i)
                        .expect("No value set in a WastValue::i32 argument!"),
                );
            }
            result.push_str(&format!("  BL {}\n", field));

            // Set the actual value (the result of the function) as the 3rd argument of
            // assert_equal.
            result.push_str("  MOV R2, R0\n");
            // Set the expected value as the 2nd argument of assert_equal.
            // Support multi-value-return?
            result.push_str(
                &expected[0]
                    .to_movs_to(1)
                    .expect("No value set in a WastValue::i32 expected value!"),
            );
            // Set the line number as the first argument.
            result.push_str(&asm::to_movs_to(0, line));
            result.push_str("  BL assert_equal\n");
        }
    }
    result
}

#[test]
fn test_i32() {
    run_tests_in("tmp/data/i32.json");
}
