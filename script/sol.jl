sol_path = "data/AlLiFeO/sol.txt"
f = open(sol_path, "r")

t = split(read(f, String), "\n")


ttt = []
for i in eachindex(t)
    tt = split(t[i], "=")[end]
    push!(ttt, vcat(["$(i)"], split(tt, ",")))
end

ttt = ttt[1:end-1]

sol = filter(x->parse(Float64, x[6])==0, ttt)

new_sol = String[]
for i in sol
    n = i[1]
    for j in 2:7
        if parse(Float64, i[j])!= 0
            n = vcat(n, ",1")
        else
            n = vcat(n, ",0")
        end
    end
    push!(new_sol, join(n))
end
new_sol = join(new_sol, "\n")

open("data/AlLiFeO/sol_new.txt", "w") do io
    write(io, new_sol)
end;
