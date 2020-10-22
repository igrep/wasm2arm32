use std::{
    env,
    fmt::Display,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process,
};

mod asm;
mod core;
mod wast_json;

use crate::core::compile;

fn main() {
    let argument = env::args().skip(1).next().expect("No wasm file specified!");
    let wasm_data = fs::read(&argument).expect(&format!("Error reading {:?}", &argument));
    compile_file(&argument, &wasm_data);
}

#[test]
fn test_example() {
    let path = "tmp/data/example.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    compile_file(path, &wasm_data);
}

#[test]
fn test_call() {
    let path = "tmp/data/call.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    let obj_path = compile_file(path, &wasm_data);
    let status = process::Command::new(&obj_path)
        .status()
        .expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(17));
}

#[test]
fn test_i32add() {
    let path = "tmp/data/i32add.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    let obj_path = compile_file(path, &wasm_data);
    let status = process::Command::new(&obj_path)
        .status()
        .expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(46));
}

#[test]
fn test_i32const() {
    let path = "tmp/data/i32const.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    let obj_path = compile_file(path, &wasm_data);
    let status = process::Command::new(&obj_path)
        .status()
        .expect("Failed to execute compiled object code!");
    assert_eq!(status.code(), Some(42));
}

#[test]
fn test_empty() {
    let path = "tmp/data/empty.wasm";
    let wasm_data = fs::read(path).expect("Error reading the test wasm file!");
    compile_file(path, &wasm_data);
}

pub fn compile_file<P: AsRef<Path> + Display>(path: P, wasm_data: &[u8]) -> PathBuf {
    let asm_path = path.as_ref().with_extension("s");
    {
        let mut asm_out =
            fs::File::create(&asm_path).expect(&format!("Failed to create file {:?}!", &asm_path));
        write!(asm_out, "{}", compile(wasm_data));
    }

    asm::compile_file(asm_path)
}
