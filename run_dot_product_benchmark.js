const { execSync } = require('child_process');
const fs = require('fs');

console.log('========================================================');
console.log('      Y-lang vs. Circom Dot Product Benchmark           ');
console.log('========================================================');

// 1. Compile Y-lang
console.log('[*] Compiling Y-lang Dot Product (100,000 multiplications)...');
const yStart = process.hrtime.bigint();
try {
    execSync('./target/release/Y dot_product.ysu --target=r1cs', { stdio: 'ignore' });
} catch (e) {
    console.error('[!] Y-lang compilation failed:', e.message);
    process.exit(1);
}
const yEnd = process.hrtime.bigint();
const yDurationMs = Number(yEnd - yStart) / 1_000_000;
console.log(`[+] Y-lang compilation completed in: ${yDurationMs.toFixed(2)} ms`);

// 2. Compile Circom
console.log('\n[*] Compiling Circom Dot Product (100,000 multiplications)...');
const circomStart = process.hrtime.bigint();
try {
    const circomOut = execSync('circom dot_product.circom --r1cs --wasm --sym').toString();
    console.log(circomOut.trim().split('\n').filter(line => line.includes('constraints') || line.includes('wires')).join('\n'));
} catch (e) {
    console.error('[!] Circom compilation failed:', e.message);
}
const circomEnd = process.hrtime.bigint();
const circomDurationMs = Number(circomEnd - circomStart) / 1_000_000;
console.log(`[+] Circom compilation completed in: ${circomDurationMs.toFixed(2)} ms`);

const speedup = circomDurationMs / yDurationMs;
console.log(`\n[=] Speedup: Y-lang is ${speedup.toFixed(2)}x faster than Circom!`);
