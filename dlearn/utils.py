class Params():
    def __init__(self):
        self.member_variables = {}
    
    def set_member_variables(self, **kwargs):
        # iterate over kwargs
        for key, value in kwargs.items():

            # set attr if key is a member variable and value is correct type
            if (key in self.member_variables) and\
                (type(value) == self.member_variables[key]):
                setattr(self, key, value)

    def get_params(self):
        # initialize params
        params = []

        # add each member variable
        for key in sorted(self.member_variables.keys()):
            params.append(self.member_variables[key])

        # return params
        return params
    
    def get_params_ordering(self):
        return sorted(self.member_variables.keys())
