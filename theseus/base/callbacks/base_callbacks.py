# Modified from YOLOv5 🚀 by Ultralytics, GPL-3.0 license

from typing import List, Dict, Any
from tabulate import tabulate
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

hook_names = [
    'sanitycheck',
    'on_start', 'on_finish',
    'on_epoch_start', 'on_epoch_end',

    'on_train_epoch_start', 'on_train_epoch_end',
    'on_train_batch_start', 'on_train_batch_end',
    'on_train_step',

    'on_val_epoch_start', 'on_val_epoch_end',
    'on_val_batch_start', 'on_val_batch_end',
    'on_val_step',
]

class Callbacks:
    """
    Abstract class for callbacks
    """

    def __init__(self) -> None:

        # Define the available callbacks
        self._hooks = {
            k: None for k in hook_names
        }

        self.name = self.__class__.__name__
        self.params = None
        self.self_register()

    def set_params(self, params):
        self.params = params

    def _do_register(self, name: str, func: Any, overide: bool = False) -> None:
        assert (
            name in self._hooks.keys()
        ), f"Method named '{name}' cannot be used as hook in {self.name}"

        assert (
            self._hooks[name] is None or overide
        ), f"""A hook named '{name}' has already been registered in {self._name}. 
        Please specify `overwrite=True` or use another name"""

        self._hooks[name] = func

    def self_register(self):
        for func_name in dir(self):
            func = getattr(self, func_name)
            if callable(func):
              if func_name in self._hooks.keys():
                  self.register_hook(func)

    def register_hook(self, func: Any = None, prefix: str = '', overide: bool = False) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """

        if func is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class._name
                self._do_register(prefix + name, func_or_class, overide=overide)
                return func_or_class
            return deco

        # used as a function call
        name = func.__name__
        self._do_register(prefix + name, func, overide=overide)


    def get(self, name: str) -> Any:
        ret = self._hooks.get(name)
        if ret is None:
            raise KeyError(
                "Hook named '{}' has not been registered in '{}'!".format(name, self._name)
            )
        return ret
    
    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._hooks.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Hook functions of {}:\n".format(self._name) + table

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


class CallbacksList:
    """"
    Handles all registered callbacks for Hooks
    """

    def __init__(self, callbacks: List[Callbacks]):

        # Define the available callbacks
        self._callbacks = {
            k: [] for k in hook_names
        }
        self._registered_callback_names = []
        self._registered_callbacks = []
        self.params = None
        # self.stop_training = False  # set True to interrupt training
        self.register_callbacks(callbacks)

    def set_params(self, params):
        for item in self._registered_callbacks:
            item.set_params(params)

    def register_callbacks(self, callbacks: List[Callbacks]):
        """
        Register list of callbacks
        """
        # Register all callbacks
        for callback in callbacks:
            if callback.name not in self._registered_callback_names:
                for method_name, method_call in callback._hooks.items():
                    if method_call is not None:
                        self.register_action(
                            method_name, 
                            name='.'.join([callback.name, method_name]),
                            callback=method_call)
                self._registered_callback_names.append(callback.name)
                self._registered_callbacks.append(callback)
            else:
                print(f"Duplicate callback named {callback.name} found.")

    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook
        Args:
            hook        The callback hook name to register the action to
            name        The name of the action for later reference
            callback    The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})
        

    def get_registered_actions(self, hook=None):
        """"
        Returns all the registered actions by callback hook
        Args:
            hook The name of the hook to check, defaults to all
        """
        if hook:
            return self._callbacks[hook]
        
        return self._callbacks

    def run(self, hook, params: Dict=None):
        """
        Loop through the registered actions and fire all callbacks
        Args:
            hook The name of the hook to check, defaults to all
            params: dict with parameters
        """

        assert hook in self._callbacks.keys(), f"hook {hook} not found in callbacks in {self._callbacks.keys()}"

        for logger in self._callbacks[hook]:
            logger['callback'](logs=params)