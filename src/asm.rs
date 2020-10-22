use std::{
    path::{Path, PathBuf},
    process,
};

pub fn compile_file<P: AsRef<Path>>(asm_path: P) -> PathBuf {
    let asm_path = asm_path.as_ref();
    let obj_path = asm_path.with_extension("o");
    let status = process::Command::new("gcc")
        .args(&[
            "-march=armv7-a",
            "-g3",
            "-o",
            obj_path.to_string_lossy().as_ref(),
            asm_path.to_string_lossy().as_ref(),
        ])
        .status()
        .expect("Unable to run gcc command!");
    if !status.success() {
        match status.code() {
            Some(code) => panic!("gcc exited with status code: {}", code),
            None => panic!("gcc terminated by signal"),
        }
    }
    obj_path
}

pub fn to_movs_to(register_num: usize, value: u32) -> String {
    let mut result = String::new();
    // https://teratail.com/questions/102402
    if value < (1 << 7) {
        result.push_str(&format!("  MOV R{}, #{}\n", register_num, value));
    } else {
        let lower_16 = value & 0x0000FFFF;
        result.push_str(&format!("  MOVW R{}, #{}\n", register_num, lower_16));

        let upper_16 = (value & 0xFFFF0000) >> 16;
        result.push_str(&format!("  MOVT R{}, #{}\n", register_num, upper_16));
    }
    result
}
