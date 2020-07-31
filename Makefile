.PHONY: test clean

CARGO_TEST_ARGS :=

test: data/example.wasm data/i32const.wasm data/empty.wasm data/i32add.wasm data/call.wasm
	cargo test --target armv7-unknown-linux-gnueabihf -- $(CARGO_TEST_ARGS)

data/%.wasm: data/%.wat
	wat2wasm -o $@ $<
	cat $@ | ssh "$(SSH_RASPI_USER_HOST)" "mkdir -p ~/tmp/data && cat > ~/tmp/data/$(@F)"

clean:
	rm data/*.wasm
	ssh "$(SSH_RASPI_USER_HOST)" "rm -f ~/tmp/data/* ~/tmp/wasm2arm-test"
