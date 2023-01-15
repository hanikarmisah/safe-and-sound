
from ovmsclient import make_grpc_client, ModelNotFoundError

class ovms_helper():
    def __init__(self, model_name, inference_url):
        self.model_name = model_name
        self.stub = make_grpc_client(inference_url)
        is_model_ready, self.model_version = self._check_model_ready()
        self.ovms_model_metadata = self.stub.get_model_metadata(model_name=self.model_name, model_version=self.model_version)
        self.input_name = 'images'
        self.output_name = 'output'

    def _check_model_ready(self):
        model_status, model_version, is_available = None, None, False
        try:
            model_status = self.stub.get_model_status(model_name=self.model_name)
        except ModelNotFoundError as err:
            raise ModelNotFoundException(f"Model {self.model_name} Not Found in OVMS Inference Server\nOVMS Exception: {str(err)}")
        except grpc.RpcError as err:
            status_code = err.code()
            if status_code == grpc.StatusCode.NOT_FOUND:
                raise ModelNotFoundException(f"Model {model_name} Not Found in OVMS Inference Server\nOVMS Exception: {str(err)}")
            else:
                raise ModelNotAvailableException(str(err))
        except Exception as err:
            """ConnectionError, TimeoutError, ..."""
            raise ModelNotAvailableException(str(err))

        model_version = max(model_status.keys())
        version_status = model_status[model_version]
        if version_status['state'] == 'AVAILABLE':
            return True, model_version

    def _predict(self, preprocessed_frame):
        inputs = {self.input_name: preprocessed_frame}
        result = self.stub.predict(inputs=inputs, model_name=self.model_name, model_version=self.model_version)
        return result