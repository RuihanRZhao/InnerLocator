from flask import Flask, request, jsonify

Network = Flask(__name__)

from device import Router, Target, Endpoint

test_router = Router(
    "router001",
    (0, 0, 0),
    "Ryen"
)
test_router.scan_devices(
    [
        Router("router002",(2,-2,1), "Ryen"),
        Target("OBJ001", (0, 1, 0), "Ryen"),
        Target("OBJ002", (1, 0, 1), "Valder"),
        Target("OBJ003", (2, 0, 1), "Ryen"),
        Endpoint("IOS001", (-1, 1, -1), "Ryen")
    ]
)
test_router.get_location_relationship()




@Network.route('/Router', methods=['GET'])
def get_routers():
    output = []
    for i in test_router.post_register_device():
        if i["type"] == "Router":
            output.append(i)
    return jsonify(output)


@Network.route('/train/location/all', methods=['GET'])
def get_all_locations():
    return jsonify(test_router.post_accurate_location_information())


@Network.route('/train/relationship/all', methods=['GET'])
def get_all_relationships():
    return jsonify(test_router.post_location_relationship())


if __name__ == '__main__':
    Network.run(debug=True, port=5000)
