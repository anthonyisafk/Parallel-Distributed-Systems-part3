using CSV, DataFrames
using PlotlyJS

v0_results = DataFrame(CSV.File("../results/v0_results.txt", header = 0));
sizes = [16, 64, 256, 512, 1024];
plots = Vector{GenericTrace}(undef, 5);

# Make one scatter plot per size.
for i = 1:5
  first = 10 * (i - 1) + 1; # index of the first test of the batch.
  last = 10 * i; # index of the last test of the batch.

  plots[i] = scatter(
    x = 1:10,
    y = v0_results[first:last, 3],
    marker_size = 8,
    name = string(sizes[i]) * "x" * string(sizes[i])
  );
end

v0_plot = plot(plots);
v0_plot.plot.layout["title"] = "V0 measurements based on model size";
savefig(v0_plot, "../image/v0_plot.jpeg", width = 872, height = 654);