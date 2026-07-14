const { execSync } = require('child_process');
const fs = require('fs');

console.log('========================================================');
console.log('      Y-lang vs. Circom Compilation Speed Test          ');
console.log('========================================================');

// 1. Benchmark Y-lang Compilation
console.log('[*] Compiling Y-lang heavy circuit (1,000,000 constraint multiplications)...');
const yStart = process.hrtime.bigint();
try {
    execSync('./target/release/Y heavy_circuit.ysu --target=r1cs', { stdio: 'ignore' });
} catch (e) {
    console.error('[!] Failed to compile Y circuit:', e.message);
    process.exit(1);
}
const yEnd = process.hrtime.bigint();
const yDurationMs = Number(yEnd - yStart) / 1_000_000;
console.log(`[+] Y-lang compilation completed in: ${yDurationMs.toFixed(2)} ms`);

// Check if circom is installed
let hasCircom = false;
try {
    execSync('circom --version', { stdio: 'ignore' });
    hasCircom = true;
} catch (e) {
    console.log('\n[!] circom compiler is not installed on this system. Skipping Circom speed test.');
}

if (hasCircom) {
    console.log('\n[*] Compiling Circom heavy circuit (1,000,000 constraints)...');
    const circomStart = process.hrtime.bigint();
    try {
        execSync('circom heavy_circuit.circom --r1cs --wasm --sym', { stdio: 'ignore' });
    } catch (e) {
        console.error('[!] Failed to compile Circom circuit:', e.message);
    }
    const circomEnd = process.hrtime.bigint();
    const circomDurationMs = Number(circomEnd - circomStart) / 1_000_000;
    console.log(`[+] Circom compilation completed in: ${circomDurationMs.toFixed(2)} ms`);
    
    const speedup = circomDurationMs / yDurationMs;
    console.log(`\n[=] Speedup: Y-lang is ${speedup.toFixed(2)}x faster than Circom!`);
}

// Check size of the generated binary files
if (fs.existsSync('heavy_circuit.r1cs')) {
    const stats = fs.statSync('heavy_circuit.r1cs');
    console.log(`\n[+] Generated R1CS binary size: ${stats.size} bytes`);
}
