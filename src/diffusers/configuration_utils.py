# coding=utf-8
"""ConfigMixin base class and utilities."""

import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from huggingface_hub import DDUFEntry, create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from typing_extensions import Self

from . import __version__
from .utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)

logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

class ConfigMixin:
    class and utilities."""

import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from huggingface_hub import DDUFEntry, create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from typing_extensions import Self

from . import __version__
from .utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)

logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

class ConfigMixin:
    
    class and utilities."""

import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from huggingface_hub import DDUFEntry, create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from typing_extensions import Self

from . import __version__
from .utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)

logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

class ConfigMixin:
    class and utilities."""

import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from huggingface_hub import DDUFEntry, create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from typing_extensions import Self

from . import __version__
from .utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)

logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

class ConfigMixin:


    config_name = None
    ignore_for_config = []
    has_compatibles = False

    _deprecated_kwargs = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:
        class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:
        
        class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:
        class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:


        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)
            return self._internal_dict[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id
            subfolder = kwargs.pop("subfolder", None)

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
                subfolder=subfolder,
            )

    @classmethod
    def from_config(
        cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs
    ) -> Union[Self, Tuple[Self, Dict[str, Any]]]:

        Instantiate a Python class from a config dictionary.
                `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
                overwrite the same named arguments in `config`.
