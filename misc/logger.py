import logging

def get_logger(cfg) -> logging.Logger:
    """ configures the logger

    Returns:
        logging.Logger
    """
    logger = logging.getLogger('default_logger')
    # create console handler
    ch = logging.StreamHandler()
    
    match cfg.get('logging_level'):
        case 'debug':
            logger.setLevel(logging.DEBUG)
            ch.setLevel(logging.DEBUG)
        case 'warning':
            logger.setLevel(logging.WARNING)
            ch.setLevel(logging.WARNING)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    
    return logger
