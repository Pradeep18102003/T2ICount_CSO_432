# Update load_model_from_config function to set weights_only=False in torch.load

def load_model_from_config(config):
    model = ...  # existing loading logic
    state_dict = torch.load(config['path'], weights_only=False)  # Modified here
    model.load_state_dict(state_dict)
    return model
