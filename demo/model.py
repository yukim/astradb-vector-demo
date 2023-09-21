class Product:
    def __init__(self, **kwargs):
        self.product_id = kwargs['product_id']
        self.product_name = kwargs['product_name']
        self.brand = kwargs['brand']
        self.short_description = kwargs['short_description']
        self.long_description = kwargs['long_description']
        self.image = kwargs['image']
        self.specs = kwargs['specs']
    