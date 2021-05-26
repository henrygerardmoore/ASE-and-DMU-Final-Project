using POMDPModelTools
using Flux
using POMDPs
using CommonRLInterface
using CommonRLInterface.Wrappers: QuickWrapper
using Distributions
using Random

include("pitch_simulation.jl")

mutable struct battingMDP <: MDP{Array{Float64,1}, Symbol}
    # state is (time, (strikes, balls), 13x1 dynamical state space)
    # actions are defined below and represent the nine combinations of:
    # forward back and neutral for rotation and up down and neutral for vertical motion
    γ::Float64
    integrator # DE integrator
    alpha_max::Float64 # max rotational accel
    a_max::Float64 # max vertical accel
    dt::Float64 # timestep
    effect_list::Array{String,1} # list of pitch effects
    pitches::pitch_ics
    strikezone::strikezone_params
    rng::MersenneTwister
    count::Tuple{Int64,Int64}
end
function parse_outcome(effect_list, solution)
    if "hit" in effect_list
        if solution.u[end][1] > 0 && solution.u[end][1] > abs(solution.u[end][2])
            return "hit", 8*sqrt(solution.u[end][1]^2 + solution.u[end][2]^2)
        else
            return "foul", 0
        end
    else
        if "strike" in effect_list
            return "strike", 0
        elseif "swung" in effect_list
            return "strike", 0
        else
            return "ball", 0
        end
    end
end

function reward_count(pitch_count, outcome, solution)
    balls = pitch_count[1]
    strikes = pitch_count[2]

    if outcome[1] == "hit"
        return clamp(outcome[2], 0, 800), (-1., -1.) # a hit
    end

    if outcome[1] == "ball"
        if balls == 3.
            return 200, (balls+1, strikes) # a walk
        else
            return 0, (balls+1, strikes)
        end
    end

    if outcome[1] == "strike"
        if strikes == 2.
            return -400, (balls, strikes+1) # a strike-out
        else
            return 0, (balls, strikes+1)
        end
    end

    if outcome[1] == "foul"
        if strikes == 2.
            return 0, (balls, 2.) # can't strike out on foul ball
        else
            return 0, (balls, strikes+1)
        end
    end
end

function eval_Q(N, mdp, Q)
    a = [:ub, :un, :uf, :nb, :nn, :nf, :db, :dn, :df]
    r_net = 0
    for i in 1:N
        # generate exp buffer
        at_bat_buffer = []
        done = false
        start_next_at_bat(mdp)
        at_bat_outcome = nothing
        while !done
            inplay = true
            while inplay
                #print("\rt: " * string(mdp.integrator.t) * " z: " * string(mdp.integrator.u[3]))
                s = observe(mdp)
                a_ind = argmax([Q(s)[ai] for ai in 1:9])
                result = take_action(mdp, s, a[a_ind], mdp.rng)
                sp = result.sp
                r = result.r
                r_net += r
                if "landed" in mdp.effect_list
                    #println(mdp.effect_list)
                    empty!(mdp.effect_list)
                    push!(mdp.effect_list, "pitched")
                    inplay = false
                end
                done = isterminal(mdp)
            end
        end
    end
    return r_net/N
end

function DDQN(N1, c, mdp::battingMDP, outfilename; Q=nothing, ϵ_heur=0.0, ϵ_pol=1.0, learning_rate=0.001)
    a = [:ub, :un, :uf, :nb, :nn, :nf, :db, :dn, :df]
    if Q == nothing
        # initialize Q if it wasn't passed in
        Q = Chain(Dense(13, 11, relu), Dense(11, 11, relu), Dense(11, length(a)))
    end
    Qalt = deepcopy(Q)
    #try
        for i in 1:N1
            println("N = " * string(i))
            if mod(i, 2000) == 0
                serialize("temp"* string(i) *".dat", Q)
                print("Evaluating Q...")
                e = eval_Q(100, mdp, Q)
                println(" done! Average score is " * string(e))
                if e > 0
                    serialize("champ" * string(ceil(e)) * ".dat", Q)
                end
                rm("temp"* string(i) *".dat")
            end
            # print("At bat " * string(i) * " of " * string(N1) * "...")

            if mod(i, c) == 0
                serialize(outfilename, Q)
            end

            # generate exp buffer
            at_bat_buffer = []
            done = false
            start_next_at_bat(mdp)
            at_bat_outcome = nothing
            while !done
                if rand() > 0.5
                    temp = Q
                    Q = Qalt
                    Qalt = temp
                end
                inplay = true
                pitch_buffer = []
                outcome = nothing
                action_select_ind = -1
                r = rand()
                # select strategy for this pitch
                if r <= ϵ_heur
                    action_select_ind = 1
                elseif r <= ϵ_heur + ϵ_pol
                    action_select_ind = 2
                else
                    action_select_ind = 3
                end
                while inplay
                    if mdp.integrator.t > 30
                        break
                    end
                    s = observe(mdp)
                    a_ind = 5
                    if action_select_ind == 1
                        a_ind = heuristic_policy(s, mdp)
                    elseif action_select_ind == 2
                        a_ind = argmax([Q(s)[ai] for ai in 1:9])
                    elseif action_select_ind == 3
                        a_ind = rand(1:9) # chooses a random action
                    end
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
            # if mdp.count == (-1, -1)
            #     at_bat_outcome = "hit"
            #     d = sqrt(at_bat_buffer[end][1][end][4][4]^2 + at_bat_buffer[end][1][end][4][5]^2)
            #     print(" hit "); print(d); println(" meters")
            # elseif mdp.count[1] == 4
            #     at_bat_outcome = "walk"
            #     print(" walk "); println(mdp.count)
            # elseif mdp.count[2] == 3
            #     at_bat_outcome = "strikeout"
            #     print(" strikeout "); println(mdp.count)
            # end

            data = form_exp_buffer(at_bat_buffer, at_bat_outcome, mdp)
            Flux.Optimise.train!((s, a_ind, r, sp, done)->loss_function_DDQN(s, a_ind, r, sp, done, Q, mdp, Qalt), Flux.params(Q), data, ADAM(learning_rate))
            if isfile("end")
                rm("end")
                println("Step " * string(i) * ", ending process...")
                break # if we put the end file, we want it to end (absolutely terrible way to do this but keyboard interrupts are dysfunctional in Juno it seems)
            end

        end
        return Q, nothing
    #catch Exception
        #println("Exception")
        # allows to manually stop learning
        #return Q, Exception
    #end
end

function bonus_reward(s, strikezone)
    r = 0
    dz = abs(clamp(s[3],strikezone.b,strikezone.t) - s[11])
    if s[1] > 0 && abs(s[2]) < strikezone.w/2 && strikezone.b <= s[3] && s[3] <= strikezone.t
        t_desired = abs(s[4]/s[1])
        omega_des = (pi/2 - s[10]) / t_desired
        d_omega = abs(s[12] - omega_des)
        r += clamp(1/d_omega, 0., 20.)
    end
    r += clamp(1/dz, 0., 20.)
    return r
end

function form_exp_buffer(at_bat_buffer, at_bat_outcome, mdp)
    # selects pertinent pitches and forms the exp buffer from those
    exp_buffer = []
    reward = at_bat_buffer[end][1][end][3] # reward of last pitch in at bat buffer is total reward
    if at_bat_outcome == "hit"
        # hit, just get the pitch we hit
        for pitch in at_bat_buffer
            # bugged
            if pitch[2] == "hit"
                for ind in length(pitch[1]):-1:1
                    cur_reward = (reward)
                    p = pitch[1][ind]
                    push!(exp_buffer, (p[1], p[2], p[3] + cur_reward, p[4], p[5]))
                    cur_reward = cur_reward * mdp.γ
                #    pitch[1][ind][3] += (reward / length(pitch[1])) # eligibility trace
                end
                break # only one hit, just return
            end
        end
    elseif at_bat_outcome == "walk"
        # walked, get all balls
        for pitch in at_bat_buffer
            if pitch[2] == "ball"
                cur_reward = reward / 4
                for ind in length(pitch[1]):-1:1
                    p = pitch[1][ind]
                    push!(exp_buffer, (p[1], p[2], p[3] + cur_reward, p[4], p[5]))
                    cur_reward = cur_reward * mdp.γ
                #    pitch[1][ind][3] += (reward / (length(pitch[1])*4)) # divide by 4 balls
                end
            end
        end
    elseif at_bat_outcome == "strikeout"
        # struck out, get fouls before 2 strikes and strikes
        strike_count = 0
        for pitch in at_bat_buffer
            if pitch[2] == "strike"
                cur_reward = reward / 3
                for ind in length(pitch[1]):-1:1
                    p = deepcopy(pitch[1][ind])
                    push!(exp_buffer, (p[1], p[2], p[3] + cur_reward, p[4], p[5]))
                    cur_reward = cur_reward * mdp.γ
                #    pitch[1][ind][3] += (reward / (length(pitch[1])*3)) # divide by 3 strikes
                end
                strike_count += 1
            end
            if pitch[2] == "foul" && strike_count < 2 # ignore fouls on 2 strikes, they didn't do anything
                for ind in length(pitch[1]):-1:1
                    cur_reward = reward / 3
                    p = deepcopy(pitch[1][ind])
                    push!(exp_buffer, (p[1], p[2], p[3] + cur_reward, p[4], p[5]))
                    cur_reward = cur_reward * mdp.γ
                #    pitch[1][ind][3] += (reward / (length(pitch[1]) * 3)) # divide by 3 strikes
                end
                strike_count += 1
            end
        end
    end
    return exp_buffer
end

function loss_function_DDQN(s, a_ind, r, sp, done, Q, mdp, Qalt)
    if done
        return (r - Q(s)[a_ind])^2
    else
        a_st = argmax(Q(sp))
        return (r + mdp.γ * Qalt(sp)[a_st] - Q(s)[a_ind])^2
    end
end


actions(mdp::battingMDP) = [:ub, :un, :uf, :nb, :nn, :nf, :db, :dn, :df];

# generative model
function take_action(mdp::battingMDP, s, a, rng)

    reward = 0
    if a == :ub
        mdp.integrator.p = [-mdp.alpha_max; mdp.a_max]
    elseif a == :un
        mdp.integrator.p = [0; mdp.a_max]
    elseif a == :uf
        mdp.integrator.p = [mdp.alpha_max; mdp.a_max]
    elseif a == :nb
        mdp.integrator.p = [-mdp.alpha_max; 0]
    elseif a == :nn
        mdp.integrator.p = [0; 0]
    elseif a == :nf
        mdp.integrator.p = [mdp.alpha_max; 0]
    elseif a == :db
        mdp.integrator.p = [-mdp.alpha_max; -mdp.a_max]
    elseif a == :dn
        mdp.integrator.p = [0; -mdp.a_max]
    elseif a == :df
        mdp.integrator.p = [mdp.alpha_max; -mdp.a_max]
    end
    if "hit" in mdp.effect_list
        step!(mdp.integrator, 100, true) # simulate to the end if we already hit the ball
    else
        step!(mdp.integrator, mdp.dt, true)
    end
    if mdp.integrator.u[3] <=0
        push!(mdp.effect_list, "landed")
    end

    # if ball didn't land yet, then the count stays the same and we get no reward other than the small negative reward for moving the bat
    new_count = (s[2], s[3])
    outcome = nothing
    if "landed" in mdp.effect_list
        outcome = parse_outcome(mdp.effect_list, mdp.integrator.sol)
        #println(outcome[1])
        reward, new_count = reward_count(mdp.count, outcome, mdp.integrator.sol)
        mdp.count = new_count
        if !is_terminal_count(new_count)
            # setup new pitch
            new_pitch = random_pitch(mdp.pitches, mdp.strikezone, mdp.rng)
            reinit!(mdp.integrator, new_pitch, erase_sol=true)
        end
        return (sp=mdp.integrator.u, r=reward, oc=outcome[1])
    end
    return (sp=mdp.integrator.u, r=reward)
end

function start_next_at_bat(mdp)
    new_pitch = random_pitch(mdp.pitches, mdp.strikezone, mdp.rng)
    empty!(mdp.effect_list)
    push!(mdp.effect_list, "pitched")
    mdp.count = (0,0)
    reinit!(mdp.integrator, new_pitch, erase_sol=true)
end


function observe(mdp::battingMDP)
    return mdp.integrator.u
end

function is_terminal_count(count)
    if count == (-1, -1)
        return true # special dummy count used for hits
    end
    if count[1] >= 4 || count[2] >=3
        return true
    end
    return false
end

function isterminal(mdp::battingMDP)
    return is_terminal_count(mdp.count)
end

function initialize_state(mdp::battingMDP)
    new_count = (0,0)
    new_pitch = random_pitch(mdp.pitches, mdp.strikezone, mdp.rng)
    empty!(mdp.effect_list)
    push!(mdp.effect_list, "pitched")
    mdp.integrator = setup_integrator(mdp.effect_list, new_pitch)
    mdp.count = new_count
end

function heuristic_policy(s, mdp)
    # actions: [:ub, :un, :uf, :nb, :nn, :nf, :db, :dn, :df]
    if s[1] < -0.5
        return 5
    end
    if "hit" in mdp.effect_list
        return 5 # if we already hit the ball, chill out
    end
    t_remaining_est = abs(s[1]/s[4]) # estimated time for the ball to get to the plate
    target_z = clamp(s[3] - t_remaining_est * s[6] - 0.5 * 9.81 * t_remaining_est^2,mdp.strikezone.b, mdp.strikezone.t) - 0.22 # get under ball
    dz = target_z - s[11] # difference between bat and ball's z val
    if t_remaining_est <1e-9
        t_remaining_est = 1e-9 # protect against dividing by too small numbers
    end
    vz = s[13]
    az_desired = 2*(dz - vz*t_remaining_est)/(t_remaining_est^2)

    # if dz > 0.02
    #     az_desired = mdp.a_max
    # elseif dz < -0.02
    #     az_desired = -mdp.a_max
    # else
    #     az_desired = 0
    # end

    idx = 3 # neutral
    if az_desired >= mdp.a_max
        idx = 0
    elseif az_desired <= -mdp.a_max
        idx = 6
    end

    target_theta = pi/2 + 0.2 # try to hit directly over plate
    dtheta = target_theta - s[10]
    alpha_desired = 3*(dtheta - s[12]*t_remaining_est)/(t_remaining_est^2)

    if !(abs(s[2]) <= mdp.strikezone.w && mdp.strikezone.b <= s[3] && s[3] <= 2.0*mdp.strikezone.t)
        alpha_desired = 0
    end

    if alpha_desired >= mdp.alpha_max
        idx += 3
    #elseif alpha_desired <= -mdp.alpha_max
        #idx += 1
    else
        idx += 2
    end

    # if rand() > 0.99
    #     println(s, idx)
    # end

    # the weird addition to idx is just selecting the corresponding action from the list
    return idx
end

function QMDP(Q, mdp)
    # skeleton for QMDP
end
