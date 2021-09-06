'''
LUCAS-related exceptions.
'''

class LucasRequestError(Exception):
    """Building request failed"""
    pass

class LucasDownloadError(Exception):
    """Data download failed"""
    pass

class LucasDataError(Exception):
    """Data processing failed"""
    pass

class LucasLoadError(Exception):
    """File open failed"""
    pass

class LucasConfigError(Exception):
    """File open failed"""
    pass
