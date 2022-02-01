using CSV, DataFrames
using Statistics
using StatsPlots

v3_results = DataFrame(CSV.File("../results/v3_results.txt", header = 0));
x = floor(Int, length(v3_results[!, 1]) / 11);
means = Vector{Float64}(undef, x);
sizes = [64, 256, 512, 1024];
blocksizes = [4, 16, 32];

for i = 1:x
  for j = 1:11
    means[i] = mean(v3_results[10(i-1)+1:10i,5]);
  end
end

names = repeat(1:4, inner = 3);
groups = repeat(["block size = " * string(b) for b in blocksizes], outer = 4);

plotlyjs()
p = groupedbar(
  names, means,
  group = groups,
  orientation = :horizontal,
  yticks = [],
  title = "V3 for sizes " * string(sizes)
);

savefig(p, "../image/v3_plot.png");