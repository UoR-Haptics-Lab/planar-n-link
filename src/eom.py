from sympy import symbols, Matrix
from sympy.physics.mechanics import dynamicsymbols, inertia, KanesMethod, Point, ReferenceFrame, RigidBody


if __name__ == '__main__':
    num_dof = 2

    # Define generalized coordinates and speeds
    generalized_positions = [dynamicsymbols(f'q{i+1}') for i in range(num_dof)]
    generalized_velocities = [dynamicsymbols(f'q{i+1}', 1) for i in range(num_dof)]
    generalized_speed = [dynamicsymbols(f'u{i+1}') for i in range(num_dof)]

    # Define link lengths and masses
    link_lengths = [symbols(f'l{i+1}') for i in range(num_dof)]
    com_link_lengths = [symbols(f'lc{i+1}') for i in range(num_dof)]
    masses = [symbols(f'm{i+1}') for i in range(num_dof)]

    # Define inertias symbols
    inertia_symbols = [symbols(f'I{i+1}xx I{i+1}yy I{i+1}zz I{i+1}xy I{i+1}yz I{i+1}zx') for i in range(num_dof)]

    # Define joint torques
    torques = [symbols(f'T{i+1}') for i in range(num_dof)]

    # Gravitational constant
    g = symbols('g')

    # Define the Newtonian reference frame
    frames = [ReferenceFrame('R0')]

    # Define reference frames for links
    for i, q in enumerate(generalized_positions):
        frames.append(frames[i].orientnew(f'R{i+1}', 'Axis', (q, frames[i].z)))

    # Define points on bodies
    points = [Point('P0')]
    for i, (link_length, frame),  in enumerate(zip(link_lengths, frames[1:])):
        points.append(points[i].locatenew(f'P{i+1}', link_length * frame.x))

    # Define centre of mass (com) points on bodies
    com_points = []
    for i, (link_length, frame, point),  in enumerate(zip(com_link_lengths, frames[1:], points[0:-1])):
        com_points.append(point.locatenew(f'Po{i+1}', link_length * frame.x))

    # Calculate velocities
    velocities = [points[0].set_vel(frames[0], 0 * frames[0].x + 0 * frames[0].y + 0 * frames[0].z)]
    for i, (point, frame) in enumerate(zip(points[1:], frames[1:])):
        velocities.append(point.v2pt_theory(points[i], frames[0], frame))

    # Define kinematic differential equations
    kinematic_differential_equations = [qp - u for (qp, u) in zip(generalized_velocities, generalized_speed)]

    # Define inertias
    inertias = []
    for inertia_symbol, frame in zip(inertia_symbols, frames):
        inertias.append(inertia(frame, inertia_symbol[0] , inertia_symbol[1], inertia_symbol[2], inertia_symbol[3], inertia_symbol[4], inertia_symbol[5]))

    # Define bodies
    bodies = []
    for i, (frame, com_point, mass, body_inertia) in enumerate(zip(frames[1:], com_points, masses, inertias)):
        bodies.append(RigidBody(f'body{i+1}', com_point, frame, mass, (body_inertia, com_point)))

    # Forces acting on the bodies
    forces = []

    # Weights
    for com_point, mass in zip(com_points, masses):
        forces.append((com_point, -mass * g * frames[0].y))

    # Torques
    for frame, torque in zip(frames[1:], torques):
        forces.append((frame, torque * frame.z))

    KM = KanesMethod(frames[0], q_ind=generalized_positions, u_ind=generalized_speed, kd_eqs=kinematic_differential_equations)
    (fr, frstar) = KM.kanes_equations(bodies, forces)

    # Mass matrix
    mass_matrix = KM.mass_matrix

    forcing = -KM.forcing

    # Coriolis term
    coriolis = forcing.subs({g: 0})

    # Gravity term
    gravity = Matrix([f.coeff(g) * g for f in forcing])

    l, m, I = symbols('l m, I')
    link_length_substitution = dict(zip(link_lengths + com_link_lengths, [l] * (len(link_lengths) + len(com_link_lengths))))
    inertia_substitution = dict(zip([inertia_symbol[2] for inertia_symbol in inertia_symbols], [I] * len(inertia_symbols)))
    mass_substitution = dict(zip(masses, [m] * len(masses)))
    dummy_variables = dict(zip(generalized_positions + generalized_velocities + generalized_speed, [f'q{i+1}' for i in range(num_dof)] + [f'qp{i+1}' for i in range(num_dof)] + [f'u{i+1}' for i in range(num_dof)]))

    print('End-effector position:')
    x = points[-1].pos_from(points[0]).express(frames[0]).subs(link_length_substitution).simplify()
    print(x.subs(dummy_variables))

    print('End-effector velocity:')
    xp = points[-1].vel(frames[0]).express(frames[0]).subs(link_length_substitution).simplify()
    print(xp.subs(dummy_variables))

    print('Mass matrix:')
    for element in mass_matrix.subs(link_length_substitution).subs(mass_substitution).subs(dummy_variables).subs(inertia_substitution):
      print(element.simplify())


    print('Coriolis:')
    C = coriolis.subs(link_length_substitution).subs(mass_substitution).subs(dummy_variables).subs(dict(zip(torques, [0] * len(torques))))
    print([Ci.simplify() for Ci in C])

    print('Gravity:')
    G = gravity.subs(link_length_substitution).subs(dummy_variables)
    print([Gi.simplify() for Gi in G])
