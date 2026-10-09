using System.Runtime.CompilerServices;
using System.Text;

internal static unsafe partial class Program
{
    // Ordinary scalar helpers use the default .NET inlining policy.
    private static bool ScanHelperUpper(long index, long seed) => ((index + seed) & 7) < 3;
    private static long ScanHelperWeight(long index) => (index & 7) + 1;

    // ASCII makes code-unit and Y byte values equal. Invalid reads return zero.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long StringScanHelper(long n, long seed)
    {
        var text = new StringBuilder();
        for (long i = 0; i < n; i++) text.Append(ScanHelperUpper(i, seed) ? 'A' : 'z');
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < text.Length; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = text.Length;
                long value = index < 0 || index >= text.Length ? 0 : text[(int)index];
                sum += value * ScanHelperWeight(i);
            }
        return sum;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long VecScanAppendHelper(long n, long seed)
    {
        var values = new List<byte>();
        for (long i = 0; i < n; i++) values.Add(ScanHelperUpper(i, seed) ? (byte)'A' : (byte)'z');
        long sum = 0;
        for (long pass = 0; pass < 8; pass++)
            for (long i = 0; i < values.Count; i++)
            {
                long index = i;
                if ((i & 127) == 0) index = -1;
                else if ((i & 127) == 1) index = values.Count;
                long value = index < 0 || index >= values.Count ? 0 : values[(int)index];
                sum += value * ScanHelperWeight(i);
            }
        return sum;
    }


}
