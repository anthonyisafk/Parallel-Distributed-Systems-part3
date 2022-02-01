using CSV, DataFrames
using Statistics
using StatsPlots

v2_results = DataFrame(CSV.File("../results/v2_results.txt", header = 0));
x = floor(Int, length(v2_results[!, 1]) / 11);
means = Vector{Float64}(undef, x);
sizes = [64, 256, 512, 1024, 2048];
blocksizes = [4, 16, 32];

for i = 1:x
  for j = 1:11
    means[i] = mean(v2_results[10(i-1)+1:10i,4]);
  end
end

names = repeat(1:5, inner = 3);
groups = repeat(["block size = " * string(b) for b in blocksizes], outer = 5);

plotlyjs()
p = groupedbar(
  names, means,
  group = groups,
  orientation = :horizontal,
  yticks = [],
  title = "V2 for sizes " * string(sizes)
);

savefig(p, "../image/v2_plot.png");