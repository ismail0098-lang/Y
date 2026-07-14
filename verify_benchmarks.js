const { execSync } = require('child_process');
const fs = require('fs');

console.log('========================================================');
const title = 'Y-lang vs. Circom Constraint Parity Verification';
console.log(title);
console.log('========================================================');

// Cleanup old files
const cleanFile = (f) => { if (fs.existsSync(f)) fs.unlinkSync(f); };
cleanFile('test_circuit_y.r1cs');
cleanFile('test_circuit_circom.r1cs');
cleanFile('test_circuit.r1cs');

// 1. Compile Y circuit to test_circuit_y.r1cs
console.log('[*] Compiling Y-lang test_circuit.ysu...');
try {
    execSync('cargo run --features zk --bin Y -- test_circuit.ysu --target=r1cs', { stdio: 'ignore' });
    fs.renameSync('test_circuit.r1cs', 'test_circuit_y.r1cs');
} catch (e) {
    console.error('[!] Failed to compile Y circuit:', e.message);
    process.exit(1);
}

// 2. Compile Circom circuit to test_circuit_circom.r1cs
let hasCircom = false;
try {
    execSync('circom --version', { stdio: 'ignore' });
    hasCircom = true;
} catch (e) {
    console.log('[!] circom compiler is not installed on this system. Skipping Circom compilation.');
}

if (hasCircom) {
    console.log('[*] Compiling Circom test_circuit.circom...');
    try {
        cleanFile('test_circuit.r1cs');
        execSync('circom test_circuit.circom -l . --r1cs --wasm --sym', { stdio: 'ignore' });
        fs.renameSync('test_circuit.r1cs', 'test_circuit_circom.r1cs');
    } catch (e) {
        console.error('[!] Failed to compile Circom circuit:', e.message);
        process.exit(1);
    }
}

// 3. Parse stats via snarkjs
function getR1csStats(filename) {
    try {
        const info = execSync(`npx --yes snarkjs r1cs info ${filename}`).toString();
        const constraintsMatch = info.match(/# of Constraints:\s+(\d+)/);
        const wiresMatch = info.match(/# of Wires:\s+(\d+)/);
        
        return {
            constraints: constraintsMatch ? parseInt(constraintsMatch[1], 10) : null,
            wires: wiresMatch ? parseInt(wiresMatch[1], 10) : null,
        };
    } catch (e) {
        console.error(`[!] Failed to run snarkjs on ${filename}:`, e.message);
        return null;
    }
}

console.log('\n[*] Extracting statistics using snarkjs...');
const yStats = getR1csStats('test_circuit_y.r1cs');
if (!yStats) {
    console.error('[!] Could not get Y-lang stats.');
    process.exit(1);
}
console.log(`[+] Y-lang: ${yStats.constraints} constraints, ${yStats.wires} wires`);

if (hasCircom) {
    const circomStats = getR1csStats('test_circuit_circom.r1cs');
    if (!circomStats) {
        console.error('[!] Could not get Circom stats.');
        process.exit(1);
    }
    console.log(`[+] Circom: ${circomStats.constraints} constraints, ${circomStats.wires} wires`);

    console.log('\n[*] Verifying parity and correctness...');
    if (yStats.constraints === 5 && yStats.wires === 8) {
        console.log('[+] PASS: Y-lang generated exactly the expected 5 constraints and 8 wires.');
    } else {
        console.error(`[!] FAIL: Y-lang constraint/wire mismatch (expected 5/8, got ${yStats.constraints}/${yStats.wires})`);
        process.exit(1);
    }

    if (circomStats.constraints === 7 && circomStats.wires === 10) {
        console.log('[+] PASS: Circom generated exactly the expected 7 constraints and 10 wires.');
    } else {
        console.error(`[!] FAIL: Circom constraint/wire mismatch (expected 7/10, got ${circomStats.constraints}/${circomStats.wires})`);
        process.exit(1);
    }
} else {
    // Fallback assert for Y-lang constraints if Circom is missing
    if (yStats.constraints === 5 && yStats.wires === 8) {
        console.log('[+] PASS: Y-lang constraint count is exactly 5 and wire count is 8.');
    } else {
        console.error(`[!] FAIL: Expected 5 constraints and 8 wires, got ${yStats.constraints}/${yStats.wires}`);
        process.exit(1);
    }
}

console.log('\n[+] Verification completed successfully.');
