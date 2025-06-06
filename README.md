# Safety-Alignment-with-Human-Directional-Feedback

## Run the experiments in the paper
To run the code of the simulated pendulum experiment:
```
cd test
python pendulum_correct_v2.py
```

To run the code of simulated drone navigation in a tube
(Assets like spatial points for the tube, neural SDF weights and learned polynomial weights are in uav_revise/assets, change dirs in files accordingly.):
```
cd uav_revise
python curve_control.py # positional constraint, polynomial
python curve_neural.py # positional constraint, neural features
python curve_control_v.py # velocity constraint added on positional constraint
```

To run the code of mujoco drone navigation game:
```
cd human_uav
python UAV_correct_human_mj_v3.py
```

To run the code of mujoco arm reaching game:
```
cd human_arm
python arm_correct_human_v2.py
```

## Control Loop Structure
```
for i in range(max_rollout_iters):
    if human_corr:
        # calculate cutting hyperplanes
        human_corr_e = np.concatenate([human_corr.reshape(-1, 1), np.zeros((u_dim * (Horizon - 1), 1))])
        h, b, h_phi, b_phi = hb_calculator.calc_planes(learned_theta,
                                                       controller.opt_traj,
                                                       human_corr=human_corr_e,
                                                       target_x=target_x)
        # solve for MVE, update theta
        mve_calc.add_constraint(h, b[0])
        mve_calc.add_constraint(h_phi, b_phi[0])
        try:
            learned_theta, C = mve_calc.solve()
        except:
            ...
    # constrained MPC    
    x = arm_env.get_curr_state()
    u = controller.control(x, weights=learned_theta, target_x=target_x)


```