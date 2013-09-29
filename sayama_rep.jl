require("discussion")

function landscape(resolution, dimensions, utility_fn)
	aspects = linspace(0, 1, int(1/resolution))
	points = combinations(aspects, dimensions)
	return map(x -> (x, utility_fn(x)), points)
end

d = Discussion()
init!(d, None)
do_discussion!(d, d.players)

d2 = Discussion()
d2.num_memory = 5
init!(d2, None)
do_discussion!(d2, d2.players)

d3 = Discussion()
d3.num_memory = 30
init!(d3, None)
do_discussion!(d3, d3.players)