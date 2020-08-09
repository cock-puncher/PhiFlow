from phi.flow import *

domain = Domain([64, 64], boundaries=PERIODIC, box=box[0:100, 0:100])
# velocity = domain.vec_grid(Noise(2, names='vector')) * 2
# velocity = domain.vec_grid(Noise((2,)), type=StaggeredGrid) * 2
velocity = domain.vec_grid(GeometryMask(Sphere([50, 50], radius=15)), StaggeredGrid) * [2, 0] + [1, 0]


def step():
    global velocity
    velocity = diffuse(velocity, 0.1, 1)
    velocity = advect.semi_lagrangian(velocity, velocity, 1.0)
    write_sim_frame(app.directory, velocity, app.frame, 'velocity')


app = App('Burgers Equation in %dD' % len(domain.resolution), framerate=5)

step()
# velocity.at_centers()

app.add_field('Velocity', lambda: velocity)
app.step = step
show(app)
