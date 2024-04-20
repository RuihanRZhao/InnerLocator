from device import Device, Router, Target, Endpoint
from flask import Flask, request, jsonify
import random,json

class Test_Environment:
    def __init__(self):
        self.central_router = Router(
            name="Central Router",
            location=(0, 0, 0),
            owner="test environment"
        )

        self.network = Flask(__name__)

        self.device_list = [
            Router(
                name="Router01",
                location=(
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                ),
                owner="test environment"),
            Router(
                name="Router02",
                location=(
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                ),
                owner="test environment"),
            Router(
                name="Router03",
                location=(
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                ),
                owner="test environment")
        ]

        for element_number in range(15):
            if random.randint(0,10)%3 == 1 or element_number == 7:
                self.device_list.append(
                    Endpoint(
                        name=f"Phone{element_number:03d}",
                        location=(
                            random.uniform(-10, 10),
                            random.uniform(-10, 10),
                            random.uniform(-10, 10),
                        ),
                        owner="test environment" if element_number != 10 else "Not in Test Environment"
                    )
                )
            else:
                self.device_list.append(
                    Target(
                        name=f"Tag{element_number:03d}",
                        location=(
                            random.uniform(-10, 10),
                            random.uniform(-10, 10),
                            random.uniform(-10, 10),
                        ),
                        owner="test environment" if element_number != 10 else "Not in Test Environment"
                    )
                )

        self.device_data_log = [self.central_router.location_dict_generate()]
        for device in self.device_list:
            self.device_data_log.append(device.location_dict_generate())

    def start(self):
        self.central_router.scan_devices(self.device_list)
        self.central_router.get_location_relationship()

        @self.network.route('/train/location/all', methods=['GET'])
        def get_accurate_location():
            return self.device_data_log

        @self.network.route('/train/relationship/all', methods=['GET'])
        def get_noisy_relationship():
            return jsonify(self.central_router.post_location_relationship())

    def run(self):
        self.network.run(debug=True, port=5000)


if __name__ == "__main__":
    env = Test_Environment()
    env.start()
    env.run()

