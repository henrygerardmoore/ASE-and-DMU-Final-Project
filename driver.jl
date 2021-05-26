using PyCall
using PyPlot
pygui(:qt5)
using Random
using Serialization
include("pitch_simulation.jl")
include("state_estimation.jl")
include("decision_making.jl")

function test_pitch()
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    integrand = (u,p,t) -> pitch_integrand(u, baseball, physical_constants, p)
    v_offset = [0., 0., 0.]
    effect_list = ["pitched"]
    rng = MersenneTwister()
    init_pitch = random_pitch(pitches, strikezone, rng)
    solution = pitch(integrand, init_pitch, baseball, bat, strikezone, effect_list, v_offset, [0.0, 10.0], [(pi/2)/(0.5* 0.5541134277416956^2); 0])
    minx = Inf
    platey = 0
    platez = 0
    for i in 1:length(solution.u)
        #if abs(solution.u[i][1]) < 1.0e-14
        #    println(solution.u[i][1], " ", solution.u[i][2], " ", solution.u[i][3])
        #    println(solution.t[i])
        #    println(solution.u[i][10], " ", solution.u[i][11])
        #    println()
        #end
        if abs(solution.u[i][1]) < minx
            minx = solution.u[i][1]
            platey = solution.u[i][2]
            platez = solution.u[i][3]
        end
    end
    println(solution.u[end][1], " ", solution.u[end][2], " ", solution.u[end][3])
    strikex, strikey, strikez = strikezone_points(strikezone)
    plot(platey, platez, "bo")
    plot(0, (strikezone.t + strikezone.b)/2.0, "go")
    plot(strikey, strikez, "r")
    xlim([-strikezone.w-0.3, strikezone.w+0.3])
    ylim([0, 2])
    axis("equal")
    plt.show()
    println(effect_list)
    #solution = pitch(integrand, pitches.curveball)
    #plot!(solution, vars=(1,2,3), xlim=(0,18.4404), ylim=(-1,1),zlim=(0,1.5), xlabel="x", ylabel="y", zlabel="z", label="Curveball")
end
function setup_integrator(effect_list, init_pitch=nothing)
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    if init_pitch == nothing
        init_pitch = [pitches.fastball; 0; (strikezone.t + strikezone.b)/2; 0; 0]
    end
    integrand = (u,p,t) -> pitch_integrand(u, baseball, physical_constants, p)
    v_offset = [0., 0., 0.]
    return pitch_integrator(integrand, init_pitch, baseball, bat, strikezone, effect_list, v_offset)
end

function setup_est_integrator(effect_list, init_pitch=nothing)
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    if init_pitch == nothing
        init_pitch = [pitches.fastball; 0; (strikezone.t + strikezone.b)/2; 0; 0]
    end
    integrand = (u,p,t) -> pitch_est_integrand(u, baseball, physical_constants, p)
    v_offset = [0., 0., 0.]
    return estimation_integrator(integrand, init_pitch, baseball, bat, strikezone, effect_list, v_offset)
end

function test_hit()
    effect_list = ["pitched"]
    integrator = setup_integrator(effect_list)
    integrator.p = [(pi/2)/(0.5* 0.5541134277416956^2); 0] # ICs to hit a changeup (very gently)
    p_set = false
    for i in integrator
        if integrator.t > 0.53 && p_set == false
            p_set = true
            integrator.p = [(1.2*pi/2 - integrator.u[10])/(0.5* (0.5541134277416956-integrator.t)^2); 0] #0.5541134277416956
        end
    end
    println(effect_list)
    outcome = parse_outcome(effect_list, integrator.sol)
    println(outcome)
end


function DDQN_main(N=1000; filename=nothing, outfilename=nothing, eps_heur_init=0.5, eps_policy_init = 0.2, learning_rate=0.01)
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    effect_list = ["pitched"]
    rng = MersenneTwister()
    init_pitch = random_pitch(pitches, strikezone, rng)
    integrator = setup_integrator(effect_list, init_pitch)
    mdp = battingMDP(0.999, integrator, 30, 20, 0.05, effect_list, pitches, strikezone, rng, (0,0))
    initialize_state(mdp)
    Q_orig = nothing
    if outfilename==nothing
        outfilename = "temp.dat"
    end
    if filename!=nothing
        print("Loading from " * filename * "...")
        Q_orig = deserialize(filename)
        println(" done!")
    end
    C = 50
    Exception=nothing
    Q_new = nothing
    while true
        Q_new, Exception = DDQN(N, C, mdp, outfilename, Q=Q_orig, ϵ_heur=eps_heur_init, ϵ_pol=eps_policy_init, learning_rate=learning_rate)
        println("Saving...")
        println("Saved to " * outfilename)
        serialize(outfilename,Q_new)

        if Exception==nothing
            println("Ending process...")
            break
        else
            println(Exception)
        end

        if isfile("actually_please_end")
            rm("actually_please_end")
            println("Ending process...")
            break # if we put the end file, we want it to end (absolutely terrible way to do this but keyboard interrupts are dysfunctional in Juno it seems)
        end
    end

    if Exception != nothing
        error(Exception)
    end
    return mdp, Q_new
end

function main_est(Ns=100)
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    effect_list = ["pitched"]
    rng = MersenneTwister()
    init_pitch = random_pitch(pitches, strikezone, rng)[1:9]
    integrator = setup_est_integrator(effect_list, init_pitch)
    mdp = battingMDP(0.999, integrator, 30, 20, 0.05, effect_list, pitches, strikezone, rng, (0,0))
    initialize_state(mdp)
    mdp.integrator = integrator
    r_noise_mag = 1
    theta_noise_mag = 0.1
    phi_noise_mag = theta_noise_mag
    head_position = [0, -1, 1.6]
    pf = particle_filter(head_position, Ns, [], [], r_noise_mag, theta_noise_mag, phi_noise_mag)
    init_particles(pf, mdp)

    # run timed step once then reset to get rid of first run slowness
    step_forward(pf, mdp)
    resample(pf)
    init_particles(pf, mdp)

    x_true_mat = []
    x_est_mat = []
    error_mat = []
    t_est_mat = [0.]
    t_mat = []
    N_ess_mat = []
    while mdp.integrator.u[1] > 0 && mdp.integrator.t < 1
        N_ess = -1
        t = @elapsed begin
            N_ess = step_forward(pf, mdp)
            resample(pf)
        end

        x_est = estimate_x(pf)

        push!(t_est_mat, t + t_est_mat[end])
        push!(error_mat, norm(x_est - mdp.integrator.u))
        push!(x_true_mat, mdp.integrator.u)
        push!(x_est_mat, x_est)
        push!(t_mat, mdp.integrator.t)
        push!(N_ess_mat, N_ess)
    end
    t_est_mat = t_est_mat[2:end]
    return t_mat, t_est_mat, x_true_mat, x_est_mat, error_mat, N_ess_mat
end

function plot_state(s)
    plt.plot(s[1], s[3],"ro")
    bat_L = 1.07
    theta = pi - s[10]
    z_bat = s[11]
    plt.plot(0, z_bat,"r*")
    plt.plot(0.5*bat_L * cos(theta), z_bat,"g*")
    plt.plot(bat_L * cos(theta), z_bat,"b*")
end
function main_heur_test()
    strikezone, baseball, bat, physical_constants, pitches = setup_constants()
    effect_list = ["pitched"]
    rng = MersenneTwister()
    init_pitch = random_pitch(pitches, strikezone, rng)
    integrator = setup_integrator(effect_list, init_pitch)
    mdp = battingMDP(0.999, integrator, 30, 20, 0.05, effect_list, pitches, strikezone, rng, (0,0))
    initialize_state(mdp)
    Exception=nothing
    a = [:ub, :un, :uf, :nb, :nn, :nf, :db, :dn, :df]
    for i in 1:10
        at_bat_buffer=[]
        done = false
        start_next_at_bat(mdp)
        at_bat_outcome = nothing
        while !done
            inplay = true
            pitch_buffer = []
            outcome = nothing
            action_select_ind = -1

            while inplay
                if mdp.integrator.t > 30
                    break
                end
                s = observe(mdp)
                a_ind = heuristic_policy(s, mdp)
                result = take_action(mdp, s, a[a_ind], mdp.rng)
                sp = result.sp
                r = result.r
                # r += bonus_reward(sp, mdp.strikezone) # heuristic bonus to encourage certain conditions
                if "landed" in mdp.effect_list
                    #println(mdp.effect_list)
                    empty!(mdp.effect_list)
                    push!(mdp.effect_list, "pitched")
                    inplay = false
                    outcome = result.oc
                end
                done = isterminal(mdp)
                experience_tuple = (s, a_ind, r, sp, done)

                push!(pitch_buffer, experience_tuple)
            end
            push!(at_bat_buffer, (pitch_buffer, outcome))
        end
        if mdp.count == (-1, -1)
            at_bat_outcome = "hit"
            d = sqrt(at_bat_buffer[end][1][end][4][4]^2 + at_bat_buffer[end][1][end][4][5]^2)
            print(" hit "); print(d); println(" meters")
        elseif mdp.count[1] == 4
            at_bat_outcome = "walk"
            print(" walk "); println(mdp.count)
        elseif mdp.count[2] == 3
            at_bat_outcome = "strikeout"
            print(" strikeout "); println(mdp.count)
        end

    end
end

function plot_est(Ns=100, save=false)
    t_mat, t_est_mat, x_true_mat, x_est_mat, error_mat, N_ess_mat = main_est(Ns)

    plt.figure()
    plot(t_mat, N_ess_mat)
    xlabel("Simulation Time (s)")
    ylabel(L"N_{ess}")
    title(L"$N_{ess}$ vs. Simulation Time, N=" * string(Ns))
    if save
        savefig("Ness_N"*string(Ns)*".png")
    end

    plt.figure()
    plot(t_mat, error_mat)
    xlabel("Simulation Time (s)")
    ylabel("MMSE Error")
    title("MMSE Error vs. Simulation Time, N=" * string(Ns))
    if save
        savefig("error_N"*string(Ns)*".png")
    end

    plt.figure()
    plot(t_mat, t_est_mat)
    xlabel("Simulation Time (s)")
    ylabel("Real Time (s)")
    title("Real Time vs. Simulation Time, N=" * string(Ns))
    if save
        savefig("time_N"*string(Ns)*".png")
    end

    x_true = []
    y_true = []
    z_true = []

    x_est = []
    y_est = []
    z_est = []
    for i = 1:length(x_true_mat)
        push!(x_true, x_true_mat[i][1])
        push!(x_est, x_est_mat[i][1])

        push!(y_true, x_true_mat[i][2])
        push!(y_est, x_est_mat[i][2])

        push!(z_true, x_true_mat[i][3])
        push!(z_est, x_est_mat[i][3])
    end

    plt.figure()
    plot(t_mat, x_est - x_true)
    xlabel("Simulation Time (s)")
    ylabel("x error")
    title("Error in X estimate vs. Simulation Time, N="*string(Ns))
    if save
        savefig("x_N"*string(Ns)*".png")
    end

    plt.figure()
    plot(t_mat, y_est - y_true)
    xlabel("Simulation Time (s)")
    ylabel("y error")
    title("Error in Y estimate vs. Simulation Time, N="*string(Ns))
    if save
        savefig("y_N"*string(Ns)*".png")
    end

    plt.figure()
    plot(t_mat, z_est - z_true)
    xlabel("Simulation Time (s)")
    ylabel("z error")
    title("Error in Z estimate vs. Simulation Time, N="*string(Ns))
    if save
        savefig("z_N"*string(Ns)*".png")
    end
end

function plot_multiple()
    for Ns = [50, 100, 200]
        plot_est(Ns, true)
    end
end
