(module
  (type $t1 (func (result i32)))
  (type $t2 (func (param i32) (param i32) (result i32)))
  (func (export "main") (type $t1)
    i32.const 1
    i32.const 5
    call $f
    i32.const 2
    i32.add
  )
  (func $f (type $t2) (param $n i32) (param $m i32) (result i32)
    get_local $n
    get_local $m
    i32.add
    i32.const 9
    i32.add
  )
)
