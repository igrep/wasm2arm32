.PHONY: test clean

CARGO_TEST_ARGS :=

test: data/example.wasm data/i32const.wasm data/empty.wasm data/i32add.wasm data/call.wasm data/i32.json
	cargo test --target armv7-unknown-linux-gnueabihf -- $(CARGO_TEST_ARGS)

data/%.wasm: data/%.wat
	wat2wasm -o $@ $<
	cat $@ | ssh "$(SSH_RASPI_USER_HOST)" "mkdir -p ~/tmp/data && cat > ~/tmp/data/$(@F)"

data/%.json: data/%.wast
	wast2json -o $@ $<
	cat $@ | ssh "$(SSH_RASPI_USER_HOST)" "mkdir -p ~/tmp/data && cat > ~/tmp/data/$(@F)"
	for wasm_file in data/*.wasm ; do \
		cat "$$wasm_file" | ssh "$(SSH_RASPI_USER_HOST)" "cat > ~/tmp/$$wasm_file" ; \
	done

clean:
	rm -f data/*.wasm data/*.json
	ssh "$(SSH_RASPI_USER_HOST)" "rm -f ~/tmp/data/* ~/tmp/wasm2arm-test"

data/*.s:
	scp $(SSH_RASPI_USER_HOST):~/tmp/$@ data/
