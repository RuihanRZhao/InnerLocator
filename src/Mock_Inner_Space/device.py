

class Device:
    def __init__(self, device_name, device_location, owner):
        self.name = device_name
        self.location = device_location
        self.owner = owner
        self.type = None

class Router(Device):
    def __init__(self, name, location, owner):
        super().__init__(name, location, owner)
        self.type = "Router"
        self.distance_storage = []
        self.registered_device = []

    def scan_devices(self,devices_list):
        # scan how much device has same owner
        for device in devices_list:
            if device.owner == self.owner:
                self.registered_device.append(device)
            else:
                pass
        print(f"Router {self.name}: Detected {len(devices_list)} devices, "
              f"{len(self.registered_device)} devices registered by {self.owner}:")
        for device in self.registered_device:
            print(f"\t{device.name}: {device.type}, {device.location}")

    def get_distances(self):
        # send get req to all objects, get information back from all objects
        for number,device in enumerate(self.registered_device):
            self.distance_storage.append(
                self.distance_dict(self, device)
            )
            for to_device in self.registered_device[number+1:]:
                self.distance_storage.append(
                    self.distance_dict(device, to_device)
                )

        print(f"Initial distances loaded: {self.distance_storage}")

    def update_distance(self, device):

        def update(from_d, to_d, ori_data):
            updated = self.distance_dict(from_d, to_d)
            print(f"Update Distance data: {ori_data} >>> {updated}")
            return updated

        for number, info in enumerate(self.distance_storage):
            if info["from"] == device.name:
                for to_device in self.registered_device:
                    if info["to"] == to_device.name:
                        # Update the dictionary in place
                        self.distance_storage[number] = update(device, to_device, info)

            if info["to"] == device.name:
                if info["from"] == self.name:
                    # Update the dictionary in place
                    self.distance_storage[number] = update(self, device, info)

                for from_device in self.registered_device:
                    # Update the dictionary in place
                    if info["from"] == from_device.name:
                        self.distance_storage[number] = update(from_device, device, info)

    def distance_dict(self, from_device, to_device):
        return {
                    "from": from_device.name,
                    "from_type": from_device.type,
                    "to": to_device.name,
                    "to_type": to_device.type,
                    "distance": sum(
                        [
                            (ob_dst - rt_dst)**2 for ob_dst, rt_dst in zip(from_device.location, to_device.location)
                        ]
                    ) ** 0.5,
        }

    def post_distances(self):
        # use internet to post to endpoint devices
        return self.distance_storage


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
            Target("OBJ001", (0,1,0), "Ryen"),
            Target("OBJ002", (1,0,1), "Valder"),
            Target("OBJ003", (2,0,1), "Ryen"),
            Endpoint("IOS001", (-1,1,-1), "Ryen")
        ]
    )

    test_router.get_distances()
    test_router.registered_device[0].location = (0,2,0)
    test_router.update_distance(test_router.registered_device[0])
    print(test_router.post_distances())


