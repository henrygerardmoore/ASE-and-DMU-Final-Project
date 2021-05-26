using DifferentialEquations
using LinearAlgebra

struct baseball_params
    r
    CD
    m
end

struct bat_params
    L
    r
    origin_loc
    coeff_rest
end

struct strikezone_params
    w # width
    t # top
    b # bottom
end

function strikezone_points(strikezone)
    p1 = [0, strikezone.w/2, strikezone.b]'
    p2 = [0, strikezone.w/2, strikezone.t]'
    p3 = [0, -strikezone.w/2, strikezone.t]'
    p4 = [0, -strikezone.w/2, strikezone.b]'
    points = [p1; p2; p3; p4; p1]
    x = points[:,1]
    y = points[:,2]
    z = points[:,3]
    return x, y, z
end
struct physical_params
    g
    rho
end

struct pitch_ics
    fastball
    slider
    curveball
    changeup
end

function setup_constants()
    strikezone_w = 0.4318; # m
    strikezone_t = 1.042416; # m
    strikezone_b = 0.5334; # m
    strikezone = strikezone_params(strikezone_w, strikezone_t, strikezone_b)

    r_baseball = 0.036829957222125655 # m
    CD_baseball = 0.3
    m_baseball = 0.145; # kg
    baseball = baseball_params(r_baseball, CD_baseball, m_baseball)

    L_bat = 1.07 # m
    r_bat = 3.3147/100 # m
    origin_bat = [0, -0.5, (strikezone_t + strikezone_b)/2] # m, point about which bat rotates
    bat_coeff_rest = 1.5 # bat coefficient of restitution (using higher than 1 to represent bat being slowed by ball, which is not modeled)
    bat = bat_params(L_bat, r_bat, origin_bat, bat_coeff_rest)

    g = 9.80665 # m/s^2
    rho = 1.225 # kg/m^2
    physical_constants = physical_params(g, rho)

    dist_to_plate = 18.4404 # m
    r0 = [dist_to_plate; 0.0; 1.52]

    theta_fast = -0.78 # deg
    phi_fast = 0.0
    v_fast = 42.4688 # m/s
    v0_fastball = [-(v_fast)*cosd(theta_fast); v_fast*sind(phi_fast) ; (v_fast)*sind(theta_fast)]
    omega0_fastball = [0.0; 40*2*pi; 0.0]
    fastball = [r0; v0_fastball; omega0_fastball]

    v_curve = 34.4221 # m/s
    theta_curve = 2.812 # deg
    phi_curve = 1.583 # deg
    v0_curveball = [-v_curve*cosd(theta_curve); v_curve*sind(phi_curve);  v_curve*sind(theta_curve)]
    omega0_curveball = [0.0; -10*2*pi; 40*2*pi]
    curveball = [r0; v0_curveball; omega0_curveball]


    v_slider = 38.9 # m/s
    theta_slider = -0.195 # deg
    phi_slider = 0.79 # deg
    v0_slider = [-v_slider*cosd(theta_slider); v_slider*sind(phi_slider);  v_slider*sind(theta_slider)]
    omega0_slider = [0.0; 40*2*pi; 20*2*pi]
    slider = [r0; v0_slider; omega0_slider]

    v_changeup = 35 # m/s
    theta_changeup = 1.07 # deg
    phi_changeup = 0.0 # deg
    v0_changeup = [-v_changeup*cosd(theta_changeup); v_changeup*sind(phi_changeup);  v_changeup*sind(theta_changeup)]
    omega0_changeup = [0.0; 30*2*pi; 0.0]
    changeup = [r0; v0_changeup; omega0_changeup]

    pitches = pitch_ics(fastball, slider, curveball, changeup)

    return strikezone, baseball, bat, physical_constants, pitches
end

function is_strike(pitch_loc, strikezone)
    if abs(pitch_loc[2]) <= strikezone.width/2 && pitch_loc[3] >= strikezone.bottom_height && pitch_loc[3] <= strikezone.top_height
        return true
    end
    return false
end


function pitch_integrand(state, baseball, physical_constants, input)
    # state = [x; y; z; vx; vy; vz; omegax; omegay; omegaz; bat_theta; bat_z; bat_omega; bat_vz]
    # origin is at plate
    # x = toward pitcher's mound
    # y = left (from batter's pov)
    # z = up

    r = baseball.r
    m = baseball.m
    CD = baseball.CD
    rho = physical_constants.rho
    g = physical_constants.g

    A = pi*r^2
    I = 0.4 * m * r^2

    x = state[1]
    y = state[2]
    z = state[3]

    vx = state[4]
    vy = state[5]
    vz = state[6]

    omega = [state[7]; state[8]; state[9]]

    magnus_f_z = -16.0/3.0 * pi^2 * r^3 * omega[2]*rho*vx
    magnus_f_y = 16.0/3.0 * pi^2 * r^3 * omega[3]*rho*vx
    magnus_f_x = 0
    magnus_accel = [magnus_f_x; magnus_f_y; magnus_f_z]/m

    v = [vx; vy; vz]
    # v_wind = [0.5*randn(); 0.5*randn(); 0]
    # v_rel = v - v_wind
    v_rel = v

    drag_accel = -0.5*rho*norm(v_rel)*CD*A * v_rel / m
    gravity = [0.0; 0.0; -g]
    a = drag_accel + gravity + magnus_accel
    alpha = -200.0 * omega

    vbat = [state[12]; state[13]]
    abat = [input[1] - 2 * state[12]; input[2] - 2 * state[13]]

    dstate = [v; a; alpha; vbat; abat]
end


function pitch_est_integrand(state, baseball, physical_constants, input)
    # state = [x; y; z; vx; vy; vz; omegax; omegay; omegaz; bat_theta; bat_z; bat_omega; bat_vz]
    # origin is at plate
    # x = toward pitcher's mound
    # y = left (from batter's pov)
    # z = up

    r = baseball.r
    m = baseball.m
    CD = baseball.CD
    rho = physical_constants.rho
    g = physical_constants.g

    A = pi*r^2
    I = 0.4 * m * r^2

    x = state[1]
    y = state[2]
    z = state[3]

    vx = state[4]
    vy = state[5]
    vz = state[6]

    omega = [state[7]; state[8]; state[9]]

    magnus_f_z = -16.0/3.0 * pi^2 * r^3 * omega[2]*rho*vx
    magnus_f_y = 16.0/3.0 * pi^2 * r^3 * omega[3]*rho*vx
    magnus_f_x = 0
    magnus_accel = [magnus_f_x; magnus_f_y; magnus_f_z]/m

    v = [vx; vy; vz]
    v_wind = [0.5*randn(); 0.5*randn(); 0]
    v_rel = v - v_wind
    # v_rel = v

    drag_accel = -0.5*rho*norm(v_rel)*CD*A * v_rel / m
    gravity = [0.0; 0.0; -g]
    a = drag_accel + gravity + magnus_accel
    alpha = -200.0 * omega

    dstate = [v; a; alpha]
end

function plate_cross_condition(u, t, integrator)
    return u[1]
end

function ground_hit_condition(u, t, integrator)
    return u[3]
end

function hit_condition(u, t, integrator, baseball, bat)
    # check if bat and ball are hitting
    # this simplified collision detection ignores hitting on the end of the bat
    # hopefully that is ok for this purpose
    phi = u[10] - pi/2
    Rm = [cos(phi) -sin(phi); sin(phi) cos(phi)] # rotation matrix to transform to bat frame
    bat_loc = bat.origin_loc
    bat_loc[3] = u[11]
    ball_loc_rel = [u[1]; u[2]; u[3]] - bat_loc # ball loc relative to bat
    ball_loc_batframe = [Rm * [ball_loc_rel[1]; ball_loc_rel[2]]; ball_loc_rel[3]] # location of ball in bat frame (z is unchanged)
    if ball_loc_batframe[2] <= bat.L
        return sqrt(ball_loc_batframe[1]^2 + ball_loc_batframe[3]^2) - (bat.r + baseball.r)
    end
    return return abs(sqrt(ball_loc_batframe[1]^2 + ball_loc_batframe[3]^2) - (bat.r + baseball.r)) + 0.5
end

function hit_effect(integrator, baseball, bat, effect_list)
    # bat-ball collision
    # simplifications: no hitting the end of the bat, no friction, no spin due to collision, instantaneous collision
    #print("hit: ")
    #println(integrator.u)
    phi = integrator.u[10] - pi/2
    Rm = [cos(phi) -sin(phi); sin(phi) cos(phi)] # rotation matrix to transform to bat frame
    bat_loc = bat.origin_loc
    bat_loc[3] = integrator.u[11]
    ball_loc_rel = [integrator.u[1]; integrator.u[2]; integrator.u[3]] - bat_loc # ball loc relative to bat
    ball_loc_batframe = [Rm * [ball_loc_rel[1]; ball_loc_rel[2]]; ball_loc_rel[3]] # location of ball in bat frame (z is unchanged)
    bat_vel_batframe = [integrator.u[12] * ball_loc_batframe[2]; 0; 0] # omega * r, ignore vertical velocity in hit
    ball_vel_batframe = [Rm * [integrator.u[4]; integrator.u[5]]; integrator.u[6]] # vel of ball in bat frame
    v_rel = ball_vel_batframe - bat_vel_batframe # vel of ball relative to bat, in bat frame
    dir = [ball_loc_batframe[1]; 0; ball_loc_batframe[3]]
    dir = dir./norm(dir) # unit vector in direction of ball from center of bat
    vperp = dot(dir, v_rel)
    if  vperp > 0
        # if ball is heading away from bat we don't care about this 'collision'
        return
    end
    push!(effect_list, "hit")
    println("Hit")
    vperp = vperp * dir # perpendicular velocity vector
    vpar = v_rel - vperp
    v2_rel = -bat.coeff_rest * vperp + vpar # we let the parallel velocity stay the same
    v2 = v2_rel + bat_vel_batframe # convert velocity back to overall, not rel to bat

    Rm2 = [cos(-phi) -sin(-phi); sin(-phi) cos(-phi)] # rotation matrix to transform back
    v2 = [Rm2 * [v2[1]; v2[2]]; v2[3]] # convert velocity back to global frame
    integrator.u[4:6] = v2 # set new velocity to be what we calculated
    integrator.u[7:9] = zeros(3) # no spin and no bat movement
end

function bat_oob_condition(u, t, integrator)
    # bat is too far swung backwards or forwards
    return pi/2 + 0.01 - abs(u[10] - pi/2) # will be 0 at bat_theta ~0 or ~pi
end

function bat_stop_effect(integrator)
    if integrator.u[10] < pi/2
        integrator.u[10] = 0 # set bat position to 0
    else
        integrator.u[10] = pi # set bat position to pi
    end
    integrator.u[12] = 0 # stop the bat's rotation
end

function bat_oob_vert_condition(u, t, integrator)
    # bat is too far swung backwards or forwards
    return 0.7 - abs(u[11] - 0.75) # will be 0 at bat_z =0.05 or =1.45
end

function bat_stop_vert_effect(integrator)
    if integrator.u[11] > 0.75
        integrator.u[11] = 1.44 # set bat vert position to 1.44
    else
        integrator.u[11] = 0.06 # set bat position to pi
    end
    integrator.u[13] = 0 # stop the bat's vertical motion
end

function stop_integration(integrator, effect_list)
    #print("End location: "); println(integrator.u[1:3])
    push!(effect_list, "landed")
    terminate!(integrator)
end

function no_effect(integrator, strikezone, effect_list)
    if strikezone.b <= integrator.u[3] && strikezone.t >= integrator.u[3] && abs(integrator.u[2]) <= strikezone.w/2
        # print("strike: ")
        push!(effect_list, "strike")
    else
        # print("ball: ")
        push!(effect_list, "ball")
    end
    # println(integrator.u)
end

function bat_swing_condition(u, t, integrator)
    return pi/2 - u[10]
end

function no_effect_swing(integrator, effect_list)
    push!(effect_list, "swung")
end

function timeout_condition(u, t, integrator)
    return t - 30
end

function timeout_effect(integrator, effect_list)
    #print("End location: "); println(integrator.u[1:3])
    push!(effect_list, "timed_out")
    terminate!(integrator)
    println("Timed out")
    Base.print_matrix(stdout, integrator.u)
end

function pitch_callbacks(baseball, bat, strikezone, effect_list)
    cb_timeout = ContinuousCallback(timeout_condition, (integrator) -> timeout_effect(integrator, effect_list), save_positions=(true,false))
    cb_swing = ContinuousCallback(bat_swing_condition, nothing, (integrator) -> no_effect_swing(integrator, effect_list), save_positions=(true,false))
    cb_platecross = ContinuousCallback(plate_cross_condition, nothing, (integrator) -> no_effect(integrator, strikezone, effect_list), save_positions=(true,false))
    cb_ground = ContinuousCallback(ground_hit_condition, (integrator) -> stop_integration(integrator, effect_list), save_positions=(true,false))
    cb_bat_oob = ContinuousCallback(bat_oob_condition, bat_stop_effect, save_positions=(false, true))
    cb_bat_oob_vertical = ContinuousCallback(bat_oob_vert_condition, bat_stop_vert_effect, save_positions=(false, true))
    cb_hit = ContinuousCallback((u, t, integrator) -> hit_condition(u, t, integrator, baseball, bat), (integrator) -> hit_effect(integrator, baseball, bat, effect_list))
    return CallbackSet(cb_swing, cb_platecross, cb_ground, cb_bat_oob, cb_bat_oob_vertical, cb_hit)
end

function pitch(integrand, to_throw, baseball, bat, strikezone, effect_list, v_offset=zeros(3), tspan=[0,10.0], u=zeros(2))
    problem = ODEProblem(integrand, to_throw + [zeros(3); v_offset; zeros(7)], tspan, u)
    cb = pitch_callbacks(baseball, bat, strikezone, effect_list)
    return DifferentialEquations.solve(problem, dt=0.01, adaptive=false, callback=cb)
end

function pitch_integrator(integrand, pitch, baseball, bat, strikezone, effect_list, v_offset=zeros(3), tspan=[0,10.0], u=zeros(2))
    problem = ODEProblem(integrand, pitch + [zeros(3); v_offset; zeros(7)], tspan, u)
    cb = pitch_callbacks(baseball, bat, strikezone, effect_list)
    return DifferentialEquations.init(problem, Tsit5(), dt=0.05, adaptive=true, callback=cb, reltol=1e-2, abstol=1e-2, maxiters=1e7)
end

function est_callbacks(baseball, bat, strikezone, effect_list)
    return ContinuousCallback(ground_hit_condition, (integrator) -> stop_integration(integrator, effect_list), save_positions=(true,false))
end

function estimation_integrator(integrand, pitch, baseball, bat, strikezone, effect_list, v_offset=zeros(3), tspan=[0,10.0], u=zeros(2))
    problem = ODEProblem(integrand, pitch + [zeros(3); v_offset; zeros(3)], tspan, u)
    cb = est_callbacks(baseball, bat, strikezone, effect_list)
    return DifferentialEquations.init(problem, Tsit5(), dt=0.05, adaptive=true, callback=cb, reltol=1e-3, abstol=1e-3, maxiters=1e7)
end

function random_pitch(pitches, strikezone, rng)
    # fastball: vy offset +-0.5 vz offset +-0.6
    # slider: vy offset +-0.45 vz offset +-0.54
    # curveball: vy offset +-0.395 vz offset +-0.47
    # changeup: vy offset +-0.4 vz offset +-0.47

    r = rand(rng)
    vyr = randn(rng)
    vzr = randn(rng)
    factor = 0.0
    if r <= 0.25
        pitch = [pitches.fastball; 0; (strikezone.t + strikezone.b)/2; 0; 0]
        v_offset = factor * [0; 0.5*vyr; 0.6*vzr]
        return pitch + [zeros(3); v_offset; zeros(7)]
    elseif r <=0.5
        pitch = [pitches.slider; 0; (strikezone.t + strikezone.b)/2; 0; 0]
        v_offset = factor * [0; 0.45*vyr; 0.54*vzr]
        return pitch + [zeros(3); v_offset; zeros(7)]
    elseif r <= 0.75
        pitch = [pitches.curveball; 0; (strikezone.t + strikezone.b)/2; 0; 0]
        v_offset = factor * [0; 0.395*vyr; 0.47*vzr]
        return pitch + [zeros(3); v_offset; zeros(7)]
    else
        pitch = [pitches.fastball; 0; (strikezone.t + strikezone.b)/2; 0; 0]
        v_offset = factor * [0; 0.4*vyr; 0.47*vzr]
        return pitch + [zeros(3); v_offset; zeros(7)]
    end
end
