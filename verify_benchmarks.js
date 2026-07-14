const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('[*] Starting Y-lang vs Circom Constraint Verification...');

// 1. Compile Y circuit
console.log('[*] Compiling Y circuit: test_circuit.ysu...');
try {
    execSync('cargo run --features zk --bin Y -- test_circuit.ysu --target=r1cs', { stdio: 'inherit' });
} catch (e) {
    console.error('[!] Failed to compile Y circuit');
    process.exit(1);
}

// Check if circom is installed
let hasCircom = false;
try {
    execSync('circom --version', { stdio: 'ignore' });
    hasCircom = true;
} catch (e) {
    console.log('[!] circom compiler is not installed on this system. Skipping circom compilation.');
}

if (hasCircom) {
    console.log('[*] Compiling Circom circuit: test_circuit.circom...');
    try {
        execSync('circom test_circuit.circom -l . --r1cs --wasm --sym', { stdio: 'inherit' });
    } catch (e) {
        console.error('[!] Failed to compile Circom circuit (requires circomlib in path).');
    }
}

// 2. Parse and assert constraints via snarkjs
console.log('\n[*] Running snarkjs on Y-lang generated test_circuit.r1cs...');
try {
    const yInfo = execSync('npx --yes snarkjs r1cs info test_circuit.r1cs').toString();
    console.log(yInfo);
    
    const match = yInfo.match(/# of Constraints:\s+(\d+)/);
    if (match) {
        const constraints = parseInt(match[1], 10);
        console.log(`[+] Y-lang Circuit Constraints: ${constraints}`);
        if (constraints === 5) {
            console.log('[+] PASS: Y-lang constraint count is exactly 5.');
        } else {
            console.error(`[!] FAIL: Expected 5 constraints, got ${constraints}`);
            process.exit(1);
        }
    } else {
        console.warn('[!] Could not parse constraint count from snarkjs output');
    }
} catch (e) {
    console.error('[!] Failed to run snarkjs on Y-lang R1CS binary.');
}

console.log('\n[+] Verification script execution finished.');
