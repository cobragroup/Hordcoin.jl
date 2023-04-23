# nsb.jl

nsb_path = abspath("EntropyMaximisation/src/nsb_octave")

function call_nsb_octave(unnormalized_distr; precision = 1e-1)::Float64
    print("NSB ")
    #print("Call with $unnormalized_distr.\n")
    K = length(unnormalized_distr)
    m = maximum(unnormalized_distr)
    kx = [];
    nx = [];
    for i in 1:m
        c = count(x -> x == i, unnormalized_distr)
        if c == 0
            continue
        end
        push!(nx, i)
        push!(kx, c)
    end
    success = false
    S_nsb = 0.0
    p = Pipe()
    while (!success) 
        try
            run(pipeline(
                `octave --eval "addpath('$nsb_path');
                    E = find_nsb_entropy([$(join(kx, ' '))], [$(join(nx, ' '))], $K, $precision, 1);
                    disp(E)"`,
                stdout=p,
                stderr=devnull))
            close(p.in)
            S_nsb = parse(Float64, readchomp(pipeline(p, `tail -n 1`)))
            if isnan(S_nsb)
                throw(DomainError("NSB returned NaN. [$(join(kx, ' '))] [$(join(nx, ' '))] $K $precision 1"))
            end
            success = true
        catch e
            close(p.in)
            precision = precision * 2
            if precision > 1.6
                throw(e)
            end
            print("Retry NSB with precision $precision.\n")
            p=Pipe()
        end
    end
    return S_nsb
end