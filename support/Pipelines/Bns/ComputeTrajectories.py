# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import h5py
import numpy as np


@click.command(name="compute-trajectories")
@click.argument(
    "h5file_name",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=1,
    required=True,
)
def compute_trajectories_command(h5file_name: str):
    """Extracts the neutron star trajectories and computes the separation.

    This uses the '/ControlSystems/BnsInertialCenters.dat' data from the control
    system for the neutron star centers in code units (solar masses). It then
    computes the separation both in code units and kilometers assuming code
    units are $M_{\mathrm{sun}}=1$.

    The output is printed to screen and should be redirected into a CSV/dat
    file.
    """

    control_system_subfile = "/ControlSystems/BnsInertialCenters.dat"

    with h5py.File(h5file_name, "r") as h5file:
        dat_file = h5file.get(control_system_subfile)
        if dat_file is None:
            subfiles = available_subfiles(h5_filename, extension=".dat")
            raise ValueError(
                f"Unable to find subfile {control_system_subfile}. Known "
                f"files are:\n{subfiles}"
            )
        legend = list(dat_file.attrs["Legend"])
        control_system_data = np.asarray(dat_file)

    time = control_system_data[:, legend.index("Time")]
    x_a = control_system_data[
        :,
        legend.index("Center_A_x"),
    ]
    y_a = control_system_data[
        :,
        legend.index("Center_A_y"),
    ]
    z_a = control_system_data[
        :,
        legend.index("Center_A_z"),
    ]
    x_b = control_system_data[
        :,
        legend.index("Center_B_x"),
    ]
    y_b = control_system_data[
        :,
        legend.index("Center_B_y"),
    ]
    z_b = control_system_data[
        :,
        legend.index("Center_B_z"),
    ]
    coord_separation = control_system_data[:, 4:7] - control_system_data[:, 1:4]
    separation = np.sqrt(
        coord_separation[:, 0] ** 2
        + coord_separation[:, 1] ** 2
        + coord_separation[:, 2] ** 2
    )

    print("""
# 0: Time
# 1: x_A
# 2: y_A
# 3: z_A
# 4: x_B
# 5: y_B
# 6: z_B
# 7: Separation [M_sun]
# 8: Separation [km]""")
    m_sun_to_km = 1.47651
    for i in range(0, len(time)):
        print(
            time[i],
            x_a[i],
            y_a[i],
            z_a[i],
            x_b[i],
            y_b[i],
            z_b[i],
            separation[i],
            separation[i] * m_sun_to_km,
        )


if __name__ == "__main__":
    compute_trajectories_command(help_option_names=["-h", "--help"])
