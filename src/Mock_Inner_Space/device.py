import numpy
from zmq import DeviceType


class Device:
    def __init__(self, device_name, device_location: tuple, owner):
        self.name = device_name
        self.location = device_location
        self.owner = owner
        self.type = None

    def location_dict_generate(self):
        return {
            "name": self.name,
            "type": self.type,
            "location": self.location,
            "owner": self.owner
        }


class Router(Device):
    def __init__(self, name, location, owner):
        super().__init__(name, location, owner)
        self.type = "Router"
        self.registered_device = []

    def post_register_device(self):
        output = []

        def pack_device_info(p_device):
            return {
                "name": p_device.name,
                "type": p_device.type,
                "location": p_device.location,
                "owner": p_device.owner
            }

        output.append(pack_device_info(self))
        for device in self.registered_device:
            output.append(pack_device_info(device))

        return output

    def scan_devices(self, devices_list):
        # scan how much device has same owner
        for device in devices_list:
            if device.owner == self.owner:
                self.registered_device.append(device)
            else:
                pass
        print(f"Router {self.name} [{self.location}]: Detected {len(devices_list)} devices, "
              f"{len(self.registered_device)} devices registered by {self.owner}:")
        for device in self.registered_device:
            print(f"\t{device.name}: {device.type}, {device.location}")

    def get_location_relationship(self):
        # send get req to all objects, get information back from all objects
        def location_relationship_dict_generate(from_device, to_device):
            from_position = numpy.array(from_device.location)
            to_position = numpy.array(to_device.location)

            light_speed = 3e8
            distance = numpy.linalg.norm(from_position - to_position)
            ideal_ToA = distance / light_speed

            if from_device.type == "Router":
                ideal_AoA = {
                    "theta": numpy.arctan2(to_position[1] - from_position[1], to_position[0] - from_position[0]),
                    "phi": numpy.arcsin((to_position[2] - from_position[2]) / distance)
                }
            else:
                ideal_AoA = None

            # Add sigma error
            sigma_ToA = 1e-9
            sigma_AoA = numpy.radians(5)
            noisy_ToA = numpy.random.normal(ideal_ToA, sigma_ToA)
            if ideal_AoA is not None:
                noisy_AoA = {
                    "theta": numpy.random.normal(ideal_AoA["theta"], sigma_AoA),
                    "phi": numpy.random.normal(ideal_AoA["phi"], sigma_AoA)
                }

            else:
                noisy_AoA = None

            return {
                "from": from_device.name,
                "from_type": from_device.type,
                "to": to_device.name,
                "to_type": to_device.type,
                "ToA": noisy_ToA,
                "AoA": noisy_AoA
            }

        location_data_storage = []
        for number, device in enumerate(self.registered_device):
            location_data_storage.append(
                location_relationship_dict_generate(self, device)
            )
            for to in self.registered_device[number+1:]:
                location_data_storage.append(
                    location_relationship_dict_generate(device, to)
                )

        print(f"Initial distances loaded: {location_data_storage}")
        return location_data_storage

    def post_location_relationship(self):
        # use internet to post to endpoint devices
        return self.get_location_relationship()


class Target(Device):
    def __init__(self, name, location, owner):
        super().__init__(name, location, owner)
        self.type = "Target"


class Endpoint(Device):
    def __init__(self, name, location, owner):
        super().__init__(name, location, owner)
        self.type = "Endpoint"


if __name__ == "__main__":
    test_router = Router(
        "router001",
        (0,0,0),
        "Ryen"
    )

    test_router.scan_devices(
        [
            Router("router002", (2, -2, 1), "Ryen"),
            Target("OBJ001", (0, 1, 0), "Ryen"),
            Target("OBJ002", (1, 0, 1), "Valder"),
            Target("OBJ003", (2, 0, 1), "Ryen"),
            Endpoint("IOS001", (-1, 1, -1), "Ryen")
        ]
    )

    test_router.get_location_relationship()
    test_router.registered_device[0].location = (0,2,0)
    test_router.update_location_relationship(test_router.registered_device[0])
    print(test_router.post_location_relationship())


