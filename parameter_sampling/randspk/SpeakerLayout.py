import numpy as np

class SpeakerPosition:
    def __init__(self, azimuth=0.0, elevation=0.0, distance=1.0):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance

    def set_coords_spherical(self, az: float, el: float, dist: float):
        self.azimuth, self.elevation, self.distance = az, el, dist

    def get_coords_spherical(self) -> list[float]:
        return [self.azimuth, self.elevation, self.distance]


class SpeakerLayout:
    def __init__(
        self,
        num_speakers=8,
        radius=1.0,
        symmetry=False,
        hemisphere=True,
        min_distance=0.2,
        elevation_bias=1.0,
        random_seed=None,
    ):
        self.num_speakers = num_speakers
        self.radius = radius
        self.symmetry = symmetry
        self.hemisphere = hemisphere
        self.min_distance = min_distance
        self.elevation_bias = elevation_bias

        if random_seed is not None:
            np.random.seed(random_seed)

    # --- Utilities -----------------------------------------------------------

    @staticmethod
    def spherical_to_cartesian(azimuth, elevation, distance):
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _pairwise_distances(points_cart):
        diff = points_cart[:, np.newaxis, :] - points_cart[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _random_points_on_sphere(self, n, radius):
        az      = np.pi * np.random.rand(n)
        el      = np.arcsin(2 * np.random.rand(n) - 1)
        dist    = np.full(n, radius)
        return az, el, dist

    def _random_points_on_hemisphere(self, n, radius):
        az      = np.pi * np.random.rand(n)
        u       = np.random.rand(n) ** self.elevation_bias
        el      = np.arcsin(u)  # [0, pi/2]
        dist    = np.full(n, radius)
        return az, el, dist

    @staticmethod
    def _mirror_spherical_deg(az):
        return -az
    
    def _generate_special_odd_speaker(self, radius):
        r = np.random.rand()

        if r < 0.95:
            az, el = 0.0, 0.0
        elif r < 0.98:
            az, el = 0.0, 0.0
        else:
            az, el = 0.0, np.pi / 2
        return np.array([[az, el, radius]])

    # --- Main ---------------------------------------------------------------

    def generate_random_layout(
        self,
        num_speakers=None,
        radius=None,
        symmetry=None,
        hemisphere=None,
        mirror_axis='az',
        enforce_odd_rule=True,
        min_distance=None,
    ):
        num_speakers    = num_speakers or self.num_speakers
        radius          = radius or self.radius
        symmetry        = self.symmetry if symmetry is None else symmetry
        hemisphere      = self.hemisphere if hemisphere is None else hemisphere
        min_distance    = min_distance or self.min_distance

        generator = self._random_points_on_hemisphere if hemisphere else self._random_points_on_sphere

        def is_valid(az, el, dist):
            cart    = self.spherical_to_cartesian(az, el, dist)
            d       = self._pairwise_distances(cart)
            np.fill_diagonal(d, np.inf)
            return np.all(d > min_distance)

        for _ in range(1000):
            if symmetry:
                half                = num_speakers // 2
                az, el, dist        = generator(half, radius)
                az_m, el_m, dist_m  = np.deg2rad(self._mirror_spherical_deg(np.degrees(az))), el, dist
                
                az_full     = np.concatenate([az, az_m])
                el_full     = np.concatenate([el, el_m])
                dist_full   = np.concatenate([dist, dist_m])

                if num_speakers % 2 == 1:
                    if enforce_odd_rule:
                        special     = self._generate_special_odd_speaker(radius)
                        az_full     = np.append(az_full, special[0, 0])
                        el_full     = np.append(el_full, special[0, 1])
                        dist_full   = np.append(dist_full, special[0, 2])
                    else:
                        az_r, el_r, dist_r  = generator(1, radius)
                        az_full             = np.append(az_full, az_r)
                        el_full             = np.append(el_full, el_r)
                        dist_full           = np.append(dist_full, dist_r)
            else:
                az_full, el_full, dist_full = generator(num_speakers, radius)
                
                if num_speakers % 2 == 1 and enforce_odd_rule:
                    special = self._generate_special_odd_speaker(radius)
                    az_full[-1], el_full[-1], dist_full[-1] = special[0]

            if is_valid(az_full, el_full, dist_full):
                break
        else:
            print("Warning: could not find a valid layout after 1000 attempts.")

        sph = np.stack([az_full, el_full, dist_full], axis=-1)
        return sph

    @staticmethod
    def spherical_array_to_cartesian(sph):
        az, el, dist = sph[:, 0], sph[:, 1], sph[:, 2]
        return SpeakerLayout.spherical_to_cartesian(az, el, dist)

    @staticmethod
    def spherical_to_degrees(sph: np.ndarray) -> np.ndarray:
        sph_deg = sph.copy()
        sph_deg[:, 0] = np.degrees(sph[:, 0])
        sph_deg[:, 1] = np.degrees(sph[:, 1])
        return sph_deg
