(module
  (type $t1 (func (result i32)))
  (func (export "main") (type $t1)
    i32.const 42
    i32.const 4
    i32.add
  )
)
