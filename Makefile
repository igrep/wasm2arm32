.PHONY: test

test: data/example.wasm
	cargo test --target armv7-unknown-linux-gnueabihf -- --nocapture

data/%.wasm: data/%.wat
	wat2wasm -o $@ $<
	cat $@ | ssh "$(SSH_RASPI_USER_HOST)" "mkdir -p ~/tmp/data && cat > ~/tmp/data/$(@F)"
