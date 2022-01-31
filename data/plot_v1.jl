using CSV, DataFrames
using PlotlyJS

v1_results = DataFrame(CSV.File("../results/v1_results.txt", header = 0));
sort!(v1_results); # put v1 results into order.

sizes = [16, 64, 256, 512, 1024, 2048];
plots = Vector{GenericTrace}(undef, 6);

# Make one scatter plot per size.
for i = 1:6
  first = 10 * (i - 1) + 1; # index of the first test of the batch.
  last = 10 * i; # index of the last test of the batch.

  plots[i] = scatter(
    x = 1:10,
    y = v1_results[first:last, 3],
    marker_size = 8,
    name = string(sizes[i]) * "x" * string(sizes[i])
  );
end

v0_plot = plot(plots);
v0_plot.plot.layout["title"] = "V1 measurements based on model size";
savefig(v0_plot, "../image/v1_plot.jpeg", width = 872, height = 654);
