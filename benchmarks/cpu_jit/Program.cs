using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text;

// These method boundaries correspond to Y's host function-pointer calls.
// Pointer indexing matches Y's unchecked GlobalMemory access.
internal static unsafe partial class Program
{
    private const ulong UnsignedSeed = 0xfedcba9876543210UL;
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long IntegerBranch(long n, long seed)
    {
        long x = seed, sum = 0;
        for (long i = 0; i < n; i++)
        {
            x = x * 48271 % 2147483647;
            if ((x & 7) < 3) sum += x;
            else sum -= x;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long RecursiveFib(long n)
    {
        if (n < 2) return n;
        return RecursiveFib(n - 1) + RecursiveFib(n - 2);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double FloatRecurrence(long n, double seed)
    {
        double x = seed;
        for (long i = 0; i < n; i++)
        {
            x = x * 1.000001 + 0.0000003;
            if (x > 2.0) x -= 1.9999;
        }
        return x;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long IndexedMemory(long* data, long n, long seed)
    {
        long sum = 0;
        for (long i = 0; i < n; i++)
        {
            long index = (i * 17 + seed) & 65535;
            long value = (data[index] * 3 + 7) & 65535;
            data[index] = value;
            sum += value;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static ulong UnsignedMix(long n, ulong seed)
    {
        ulong x = seed;
        for (long i = 0; i < n; i++)
        {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x *= 2685821657736338717UL;
            x = (x << 13) | (x >> 51);
        }
        return x;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double FloatDot(double* a, double* b, long n, long seed)
    {
        double sum = 0;
        for (long i = 0; i < n; i++)
        {
            long index = (i + seed) & 65535;
            sum += a[index] * b[index];
        }
        return sum;
    }

    private static bool BumpIf(long* counters, long value)
    {
        counters[0]++;
        return (value & 7) < 3;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long ShortCircuit(long* counters, long n, long seed)
    {
        long x = seed, sum = 0;
        for (long i = 0; i < n; i++)
        {
            x = x * 48271 % 2147483647;
            if (((x & 1) == 0) && BumpIf(counters, x))
            {
                counters[1]++;
                sum += x;
            }
            if (((x & 3) == 3) || BumpIf(counters, x))
            {
                counters[2]++;
                sum -= x;
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long BinarySearch(long* data, long n, long seed)
    {
        long x = seed, sum = 0;
        for (long i = 0; i < n; i++)
        {
            x = x * 48271 % 2147483647;
            long target = x % 196608;
            long lower = 0, upper = 65535, found = -1;
            while (lower <= upper)
            {
                long middle = (lower + upper) / 2;
                long value = data[middle];
                if (value == target) { found = middle; break; }
                else if (value < target) lower = middle + 1;
                else upper = middle - 1;
            }
            sum += found;
        }
        return sum;
    }

    // ASCII makes code-unit and Y byte values equal. Invalid reads return zero.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long StringScan(long n, long seed)
    {
        var text = new StringBuilder();
        for (long i = 0; i < n; i++) text.Append(((i + seed) & 7) < 3 ? 'A' : 'z');
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < text.Length; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = text.Length;
                long value = index < 0 || index >= text.Length ? 0 : text[(int)index];
                sum += value * ((i & 7) + 1);
            }
        return sum;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long VecScanAppend(long n, long seed)
    {
        var values = new List<byte>();
        for (long i = 0; i < n; i++) values.Add(((i + seed) & 7) < 3 ? (byte)'A' : (byte)'z');
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < values.Count; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = values.Count;
                long value = index < 0 || index >= values.Count ? 0 : values[(int)index];
                sum += value * ((i & 7) + 1);
            }
        return sum;
    }


    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long VecDynamicByte(long n, long seed, int elementSize)
    {
        if (elementSize != sizeof(byte)) throw new ArgumentOutOfRangeException(nameof(elementSize));
        var values = new List<byte>();
        for (long i = 0; i < n; i++) values.Add(((i + seed) & 7) < 3 ? (byte)'A' : (byte)'z');
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < values.Count; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = values.Count;
                long value = index < 0 || index >= values.Count ? 0 : values[(int)index];
                sum += value * ((i & 7) + 1);
            }
        return sum + values.Count;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long VecDynamicI64(long n, long seed, int elementSize)
    {
        if (elementSize != sizeof(long)) throw new ArgumentOutOfRangeException(nameof(elementSize));
        var values = new List<long>();
        for (long i = 0; i < n; i++)
            values.Add(((i * 7919 + seed) & 2147483647) + ((i * 17 + seed) & 65535) * 4294967296);
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < values.Count; i++) sum += values[(int)i] * ((i & 7) + 1);
        return sum + values.Count;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long StringBulkAppend(long n, long seed)
    {
        var text = new StringBuilder();
        var chunk = new StringBuilder();
        for (long i = 0; i < 32; i++) chunk.Append(((i + seed) & 7) < 3 ? 'A' : 'z');
        for (long i = 0; i < n; i++) text.Append(chunk);
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < text.Length; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = text.Length;
                long value = index < 0 || index >= text.Length ? 0 : text[(int)index];
                sum += value * ((i & 7) + 1);
            }
        return sum + text.Length;
    }

    private static void Reset(long[] data)
    {
        for (int i = 0; i < data.Length; i++) data[i] = (i * 13 + 5) & 1023;
    }

    private static string HashMemory(long[] data)
    {
        ulong hash = 14695981039346656037UL;
        foreach (long value in data) hash = unchecked((hash ^ (ulong)value) * 1099511628211UL);
        return hash.ToString();
    }

    private static long Elapsed(long start) => Stopwatch.GetTimestamp() - start;
    private static double Nanoseconds(long ticks) => ticks * (1e9 / Stopwatch.Frequency);

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length != 12 && args.Length != 14 && args.Length != 17) throw new ArgumentException(
                "usage: CSharpBench CALLS WARMUP INTEGER_N FIB_N FLOAT_N MEMORY_N UNSIGNED_N DOT_N LOGICAL_N SEARCH_N [STRING_N VEC_N [DYNAMIC_BYTE_N DYNAMIC_I64_N BULK_N]] COLD_ONLY SUITE");
            int calls = int.Parse(args[0]), warmup = int.Parse(args[1]);
            long integerN = long.Parse(args[2]), fibN = long.Parse(args[3]);
            long floatN = long.Parse(args[4]), memoryN = long.Parse(args[5]);
            long unsignedN = long.Parse(args[6]), dotN = long.Parse(args[7]);
            long logicalN = long.Parse(args[8]), searchN = long.Parse(args[9]);
            int tail = args.Length - 2;
            long stringN = tail >= 12 ? long.Parse(args[10]) : 16384;
            long vecN = tail >= 12 ? long.Parse(args[11]) : 16384;
            long dynamicByteN = tail == 15 ? long.Parse(args[12]) : 16384;
            long dynamicI64N = tail == 15 ? long.Parse(args[13]) : 16384;
            long bulkN = tail == 15 ? long.Parse(args[14]) : 512;
            bool coldOnly = args[tail] == "1", helpersSuite = args[tail + 1] == "helpers";
            bool copiesSuite = helpersSuite || args[tail + 1] == "copies";
            bool runtimeSuite = copiesSuite || args[tail + 1] == "runtime";
            bool expanded = args[tail + 1] switch {
                "expanded" or "runtime" or "copies" or "helpers" => true, "original" => false,
                _ => throw new ArgumentException("suite must be original, expanded, runtime, copies or helpers") };
            if (calls < 1 || warmup < 0 || integerN < 1 || fibN < 2 || fibN > 40
                || floatN < 1 || memoryN < 1 || unsignedN < 1 || dotN < 1
                || logicalN < 1 || searchN < 1 || stringN < 1 || vecN < 1 || dynamicByteN < 1 || dynamicI64N < 1 || bulkN < 1) throw new ArgumentOutOfRangeException();

            // Reflection lookup and clock initialization happen before the timer.
            List<string> methodNames = [nameof(IntegerBranch), nameof(RecursiveFib), nameof(FloatRecurrence), nameof(IndexedMemory)];
            if (expanded) methodNames.AddRange([nameof(UnsignedMix), nameof(FloatDot), nameof(BumpIf), nameof(ShortCircuit), nameof(BinarySearch)]);
            if (runtimeSuite) methodNames.AddRange([nameof(StringScan), nameof(VecScanAppend)]);
            if (copiesSuite) methodNames.AddRange([nameof(VecDynamicByte), nameof(VecDynamicI64), nameof(StringBulkAppend)]);
            if (helpersSuite) methodNames.AddRange([nameof(StringScanHelper), nameof(VecScanAppendHelper), nameof(ScanHelperUpper), nameof(ScanHelperWeight)]);
            MethodInfo[] methods = methodNames.Select(name => typeof(Program).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!).ToArray();
            Nanoseconds(Elapsed(Stopwatch.GetTimestamp()));
            long started = Stopwatch.GetTimestamp();
            foreach (MethodInfo method in methods) RuntimeHelpers.PrepareMethod(method.MethodHandle);
            double compileNs = Nanoseconds(Elapsed(started));

            long[] data = new long[65536];
            Reset(data);
            double[] a = new double[expanded ? 65536 : 0], b = new double[expanded ? 65536 : 0];
            long[] sorted = new long[expanded ? 65536 : 0], counters = new long[3];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = ((i * 17 + 3) & 1023) / 1024.0;
                b[i] = ((i * 29 + 7) & 1023) / 2048.0;
                sorted[i] = i * 3 + 1;
            }
            started = Stopwatch.GetTimestamp();
            long firstInteger = IntegerBranch(integerN, 123);
            long firstFib = RecursiveFib(fibN);
            double firstFloat = FloatRecurrence(floatN, 0.5);
            long firstMemory;
            fixed (long* pointer = data) firstMemory = IndexedMemory(pointer, memoryN, 123);
            ulong firstUnsigned = 0;
            double firstDot = 0;
            long firstLogical = 0, firstSearch = 0;
            if (expanded)
            {
                firstUnsigned = UnsignedMix(unsignedN, UnsignedSeed);
                fixed (double* ap = a, bp = b) firstDot = FloatDot(ap, bp, dotN, 123);
                fixed (long* cp = counters, sp = sorted)
                {
                    firstLogical = ShortCircuit(cp, logicalN, 123);
                    firstSearch = BinarySearch(sp, searchN, 123);
                }
            }
            long firstString = runtimeSuite ? StringScan(stringN, 123) : 0;
            long firstVec = runtimeSuite ? VecScanAppend(vecN, 123) : 0;
            long firstDynamicByte = copiesSuite ? VecDynamicByte(dynamicByteN, 123, 1) : 0;
            long firstDynamicI64 = copiesSuite ? VecDynamicI64(dynamicI64N, 123, 8) : 0;
            long firstBulk = copiesSuite ? StringBulkAppend(bulkN, 123) : 0;
            long firstHelperString = helpersSuite ? StringScanHelper(stringN, 123) : 0;
            long firstHelperVec = helpersSuite ? VecScanAppendHelper(vecN, 123) : 0;
            double firstCallNs = Nanoseconds(Elapsed(started));
            string firstMemoryHash = HashMemory(data);
            long[] firstCounters = (long[])counters.Clone();

            if (coldOnly)
            {
                Console.WriteLine(JsonSerializer.Serialize(new {
                    engine = "csharp", compile_ns = compileNs, first_call_ns = firstCallNs,
                    first_integer = firstInteger.ToString(), first_fib = firstFib.ToString(),
                    first_float = firstFloat, first_memory = firstMemory.ToString(),
                    first_memory_hash = firstMemoryHash,
                    first_unsigned = firstUnsigned.ToString(), first_dot = firstDot,
                    first_logical = firstLogical.ToString(), first_search = firstSearch.ToString(),
                    first_counters = firstCounters, first_string = firstString.ToString(), first_vec = firstVec.ToString(),
                    first_dynamic_byte = firstDynamicByte.ToString(), first_dynamic_i64 = firstDynamicI64.ToString(), first_bulk = firstBulk.ToString(),
                    first_helper_string = firstHelperString.ToString(), first_helper_vec = firstHelperVec.ToString(),
                    runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
                }));
                return 0;
            }

            for (int i = 0; i < warmup; i++)
            {
                IntegerBranch(integerN, 123 + i * 17);
                RecursiveFib(fibN + i % 2);
                FloatRecurrence(floatN, 0.5 + i * 0.0001);
                fixed (long* pointer = data) IndexedMemory(pointer, memoryN, 123 + i * 17);
                if (expanded)
                {
                    UnsignedMix(unsignedN, unchecked(UnsignedSeed + (ulong)i * 17));
                    fixed (double* ap = a, bp = b) FloatDot(ap, bp, dotN, 123 + i * 17);
                    fixed (long* cp = counters, sp = sorted)
                    {
                        ShortCircuit(cp, logicalN, 123 + i * 17);
                        BinarySearch(sp, searchN, 123 + i * 17);
                    }
                }
                if (runtimeSuite)
                {
                    StringScan(stringN, 123 + i * 17);
                    VecScanAppend(vecN, 123 + i * 17);
                }
                if (copiesSuite)
                {
                    VecDynamicByte(dynamicByteN, 123 + i * 17, 1);
                    VecDynamicI64(dynamicI64N, 123 + i * 17, 8);
                    StringBulkAppend(bulkN, 123 + i * 17);
                }
                if (helpersSuite)
                {
                    StringScanHelper(stringN, 123 + i * 17);
                    VecScanAppendHelper(vecN, 123 + i * 17);
                }
            }

            var results = new List<object>();
            long[] integerOutputs = new long[calls];
            started = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++) integerOutputs[i] = IntegerBranch(integerN, 123 + i * 17);
            double integerNs = Nanoseconds(Elapsed(started));
            results.Add(new { name = "integer_branch", ns_per_call = integerNs / calls, outputs = integerOutputs });

            long[] fibOutputs = new long[calls];
            started = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++) fibOutputs[i] = RecursiveFib(fibN + i % 2);
            double fibNs = Nanoseconds(Elapsed(started));
            results.Add(new { name = "recursive_fib", ns_per_call = fibNs / calls, outputs = fibOutputs });

            double[] floatOutputs = new double[calls];
            started = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++) floatOutputs[i] = FloatRecurrence(floatN, 0.5 + i * 0.0001);
            double floatNs = Nanoseconds(Elapsed(started));
            results.Add(new { name = "float_recurrence", ns_per_call = floatNs / calls, outputs = floatOutputs });

            Reset(data);
            long[] memoryOutputs = new long[calls];
            started = Stopwatch.GetTimestamp();
            fixed (long* pointer = data)
                for (int i = 0; i < calls; i++) memoryOutputs[i] = IndexedMemory(pointer, memoryN, 123 + i * 17);
            double memoryNs = Nanoseconds(Elapsed(started));
            results.Add(new { name = "indexed_memory", ns_per_call = memoryNs / calls,
                outputs = memoryOutputs, memory_hash = HashMemory(data) });

            if (expanded)
            {
                ulong[] unsignedOutputs = new ulong[calls];
                started = Stopwatch.GetTimestamp();
                for (int i = 0; i < calls; i++) unsignedOutputs[i] = UnsignedMix(unsignedN, unchecked(UnsignedSeed + (ulong)i * 17));
                double unsignedNs = Nanoseconds(Elapsed(started));
                results.Add(new { name = "unsigned_mix", ns_per_call = unsignedNs / calls, outputs = unsignedOutputs.Select(x => x.ToString()).ToArray() });

                double[] dotOutputs = new double[calls];
                started = Stopwatch.GetTimestamp();
                fixed (double* ap = a, bp = b)
                    for (int i = 0; i < calls; i++) dotOutputs[i] = FloatDot(ap, bp, dotN, 123 + i * 17);
                double dotNs = Nanoseconds(Elapsed(started));
                results.Add(new { name = "float_dot", ns_per_call = dotNs / calls, outputs = dotOutputs });

                Array.Clear(counters);
                long[] logicalOutputs = new long[calls];
                started = Stopwatch.GetTimestamp();
                fixed (long* cp = counters)
                    for (int i = 0; i < calls; i++) logicalOutputs[i] = ShortCircuit(cp, logicalN, 123 + i * 17);
                double logicalNs = Nanoseconds(Elapsed(started));
                results.Add(new { name = "short_circuit", ns_per_call = logicalNs / calls, outputs = logicalOutputs, counters });

                long[] searchOutputs = new long[calls];
                started = Stopwatch.GetTimestamp();
                fixed (long* sp = sorted)
                    for (int i = 0; i < calls; i++) searchOutputs[i] = BinarySearch(sp, searchN, 123 + i * 17);
                double searchNs = Nanoseconds(Elapsed(started));
                results.Add(new { name = "binary_search", ns_per_call = searchNs / calls, outputs = searchOutputs });
            }

            if (runtimeSuite)
            {
                foreach (var (name, function, n) in new (string, Func<long, long, long>, long)[] {
                    ("string_scan", StringScan, stringN), ("vec_scan_append", VecScanAppend, vecN) })
                {
                    long[] outputs = new long[calls];
                    int[] collectionsBefore = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
                    long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                    started = Stopwatch.GetTimestamp();
                    for (int i = 0; i < calls; i++) outputs[i] = function(n, 123 + i * 17);
                    double duration = Nanoseconds(Elapsed(started));
                    long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                    int[] collections = [GC.CollectionCount(0) - collectionsBefore[0],
                        GC.CollectionCount(1) - collectionsBefore[1], GC.CollectionCount(2) - collectionsBefore[2]];
                    results.Add(new { name, ns_per_call = duration / calls, outputs,
                        managed_allocated_bytes = allocatedBytes, gc_collections = collections });
                }
            }

            if (copiesSuite)
            {
                foreach (var name in new[] { "vec_dynamic_byte", "vec_dynamic_i64", "string_bulk_append" })
                {
                    long[] outputs = new long[calls];
                    int[] collectionsBefore = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
                    long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                    started = Stopwatch.GetTimestamp();
                    if (name == "vec_dynamic_byte")
                        for (int i = 0; i < calls; i++) outputs[i] = VecDynamicByte(dynamicByteN, 123 + i * 17, 1);
                    else if (name == "vec_dynamic_i64")
                        for (int i = 0; i < calls; i++) outputs[i] = VecDynamicI64(dynamicI64N, 123 + i * 17, 8);
                    else
                        for (int i = 0; i < calls; i++) outputs[i] = StringBulkAppend(bulkN, 123 + i * 17);
                    double duration = Nanoseconds(Elapsed(started));
                    long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                    int[] collections = [GC.CollectionCount(0) - collectionsBefore[0],
                        GC.CollectionCount(1) - collectionsBefore[1], GC.CollectionCount(2) - collectionsBefore[2]];
                    results.Add(new { name, ns_per_call = duration / calls, outputs,
                        managed_allocated_bytes = allocatedBytes, gc_collections = collections });
                }
            }

            if (helpersSuite)
            {
                foreach (var (name, function, n) in new (string, Func<long, long, long>, long)[] {
                    ("string_scan_helper", StringScanHelper, stringN), ("vec_scan_append_helper", VecScanAppendHelper, vecN) })
                {
                    long[] outputs = new long[calls];
                    int[] collectionsBefore = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
                    long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                    started = Stopwatch.GetTimestamp();
                    for (int i = 0; i < calls; i++) outputs[i] = function(n, 123 + i * 17);
                    double duration = Nanoseconds(Elapsed(started));
                    long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                    int[] collections = [GC.CollectionCount(0) - collectionsBefore[0],
                        GC.CollectionCount(1) - collectionsBefore[1], GC.CollectionCount(2) - collectionsBefore[2]];
                    results.Add(new { name, ns_per_call = duration / calls, outputs,
                        managed_allocated_bytes = allocatedBytes, gc_collections = collections });
                }
            }

            Console.WriteLine(JsonSerializer.Serialize(new {
                engine = "csharp", compile_ns = compileNs, first_call_ns = firstCallNs,
                first_integer = firstInteger.ToString(), first_fib = firstFib.ToString(),
                first_float = firstFloat, first_memory = firstMemory.ToString(),
                first_memory_hash = firstMemoryHash, calls, warmup, results,
                first_unsigned = firstUnsigned.ToString(), first_dot = firstDot,
                first_logical = firstLogical.ToString(), first_search = firstSearch.ToString(),
                first_counters = firstCounters, first_string = firstString.ToString(), first_vec = firstVec.ToString(),
                    first_dynamic_byte = firstDynamicByte.ToString(), first_dynamic_i64 = firstDynamicI64.ToString(), first_bulk = firstBulk.ToString(),
                    first_helper_string = firstHelperString.ToString(), first_helper_vec = firstHelperVec.ToString(),
                runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
            }));
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
