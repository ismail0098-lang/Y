const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('========================================================');
console.log('    ZK Compiler Benchmark: 100,000 Constraints          ');
console.log('     (Dot Product Loop - Y vs Circom vs Noir vs Leo)    ');
console.log('========================================================\n');

function cleanDir(dirPath) {
    if (fs.existsSync(dirPath)) {
        fs.rmSync(dirPath, { recursive: true, force: true });
    }
}

function runWithStats(name, cmd, cwd = null) {
    console.log(`[*] Running ${name}...`);
    try {
        const cwdArg = cwd ? `, cwd='${cwd}'` : '';
        const pythonCmd = `python -c "import subprocess, resource, time; start = time.time(); p = subprocess.Popen('${cmd}', shell=True${cwdArg}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); p.wait(); print(f'{time.time() - start:.3f},{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.2f}')"`;
        const result = execSync(pythonCmd).toString().trim();
        const [duration, memory] = result.split(',');
        console.log(`    -> Time: ${duration}s`);
        console.log(`    -> Peak Memory: ${parseFloat(memory).toFixed(2)} MB`);
        return { duration: parseFloat(duration), memory: parseFloat(memory) };
    } catch (e) {
        console.log(`    -> Failed to run ${name}: ${e.message}`);
        return null;
    }
}

const stats = {};

// 1. Y-lang
if (fs.existsSync('dot_product.r1cs')) fs.unlinkSync('dot_product.r1cs');
stats['Y-lang'] = runWithStats('Y-lang Compiler', './target/release/Y dot_product.ysu --target=r1cs');

// 2. Circom
let hasCircom = false;
try {
    execSync('circom --version', { stdio: 'ignore' });
    hasCircom = true;
} catch (e) {}

if (hasCircom) {
    if (fs.existsSync('dot_product.r1cs')) fs.unlinkSync('dot_product.r1cs');
    stats['Circom'] = runWithStats('Circom Compiler', 'circom dot_product.circom --r1cs --wasm --sym');
}

// 3. Noir
const nargoPath = '/home/yumin/.nargo/bin/nargo';
let hasNoir = false;
try {
    execSync(`${nargoPath} --version`, { stdio: 'ignore' });
    hasNoir = true;
} catch (e) {}

if (hasNoir) {
    cleanDir(path.join(__dirname, 'noir/dot_product/target'));
    stats['Noir'] = runWithStats('Noir Compiler (Nargo)', `${nargoPath} compile --force`, 'noir/dot_product');
}

// 4. Leo
let hasLeo = false;
try {
    execSync('leo --version', { stdio: 'ignore' });
    hasLeo = true;
} catch (e) {}

if (hasLeo) {
    cleanDir(path.join(__dirname, 'leo/dot_product/build'));
    stats['Leo'] = runWithStats('Leo Compiler', 'leo build', 'leo/dot_product');
}

console.log('\n========================================================');
console.log('                  Summary Table                         ');
console.log('========================================================');
console.log(String('Compiler').padEnd(12) + ' | ' + String('Time (s)').padEnd(10) + ' | ' + String('Memory (MB)').padEnd(12) + ' | ' + String('Speedup vs Circom'));
console.log('-'.repeat(60));

const circomTime = stats['Circom'] ? stats['Circom'].duration : null;

for (const [compiler, data] of Object.entries(stats)) {
    if (!data) continue;
    const speedup = circomTime ? (circomTime / data.duration).toFixed(2) + 'x' : 'N/A';
    console.log(
        compiler.padEnd(12) + ' | ' +
        data.duration.toFixed(3).padEnd(10) + ' | ' +
        data.memory.toFixed(2).padEnd(12) + ' | ' +
        speedup
    );
}
console.log('========================================================');
