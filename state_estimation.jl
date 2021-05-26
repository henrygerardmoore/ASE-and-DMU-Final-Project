using Statistics
using LinearAlgebra
using Distributions
using StatsBase

function measurement(pf, state, no_noise=false)
    head_position = pf.head_position

    x = state[1] - head_position[1]
    y = state[2] - head_position[2]
    z = state[3] - head_position[3]

    r = sqrt(x^2 + y^2 + z^2)
    theta = acos(z/r)
    phi = atan(y,x)
    if no_noise
        return [r; theta; phi]
    else
        return [r  + pf.r_noise_mag*randn(); theta + pf.theta_noise_mag*randn(); phi + pf.phi_noise_mag*randn()]
    end
end

mutable struct particle
    integrator
end


mutable struct particle_filter
    head_position
    Ns::Int64
    particles
    weights
    r_noise_mag::Float64
    theta_noise_mag::Float64
    phi_noise_mag::Float64
end

function init_particles(pf::particle_filter, mdp)
    reinit!(mdp.integrator)
    particles = []
    weights = []
    for i = 1:pf.Ns
        pitch = random_pitch(mdp.pitches, mdp.strikezone, mdp.rng)[1:9]
        effect_list = ["pitched"]
        integrator = setup_est_integrator(effect_list, pitch)
        new_particle = particle(integrator)
        push!(particles, new_particle)
        push!(weights, 1/pf.Ns) # initialize all weights the same
    end
    pf.particles = particles
    pf.weights = weights
end

function step_forward(pf, mdp)
    # step real pitch forward
    step!(mdp.integrator, mdp.dt, true)
    x = mdp.integrator.u
    y = measurement(pf, x)
    r_pdf = Normal(y[1], pf.r_noise_mag)
    theta_pdf = Normal(y[2], pf.theta_noise_mag)
    phi_pdf = Normal(y[3], pf.phi_noise_mag)

    Threads.@threads for i = 1:pf.Ns
        step!(pf.particles[i].integrator, mdp.dt, true)
        y_part = measurement(pf, pf.particles[i].integrator.u, true)
        pf.weights[i] = pdf(r_pdf, y_part[1]) * pdf(theta_pdf, y_part[2]) * pdf(phi_pdf, y_part[3])
    end
    sum_weights = sum(pf.weights)
    pf.weights = pf.weights./sum_weights
    N_ess = 1/sum_weights^2
    return N_ess
end

function resample(pf)
    indices = sample(1:pf.Ns, Weights(pf.weights), pf.Ns)
    new_particles = deepcopy(pf.particles)
    for i in 1:pf.Ns
        new_particles[i].integrator.u = pf.particles[indices[i]].integrator.u
    end
    pf.particles = new_particles
    pf.weights[:].=1/pf.Ns
end

function estimate_x(pf)
    est = zeros(length(pf.particles[1].integrator.u))
    for i = 1:length(pf.particles)
        est = est + pf.particles[i].integrator.u * pf.weights[i]
    end
    return est
end
