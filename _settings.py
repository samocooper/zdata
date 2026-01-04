"""
Settings management for zdata.

This module provides a centralized configuration system that allows users to
customize zdata behavior through code or environment variables.
"""

from __future__ import annotations

import inspect
import os
import textwrap
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from inspect import Parameter, signature
from types import GenericAlias, NoneType
from typing import TYPE_CHECKING, Callable, NamedTuple, TypeGuard, cast

if TYPE_CHECKING:
    from typing import Any, Self


class DeprecatedOption(NamedTuple):
    """Information about a deprecated option."""

    option: str
    message: str | None
    removal_version: str | None


def _is_plain_type(obj: object) -> TypeGuard[type]:
    """Check if an object is a plain type (not a GenericAlias)."""
    return isinstance(obj, type) and not isinstance(obj, GenericAlias)


def describe(self: RegisteredOption, *, as_rst: bool = False) -> str:
    """Generate a description string for a registered option."""
    type_str = self.type.__name__ if _is_plain_type(self.type) else str(self.type)
    if as_rst:
        default_str = repr(self.default_value).replace("\\", "\\\\")
        doc = f"""\
        .. attribute:: settings.{self.option}
           :type: {type_str}
           :value: {default_str}

           {self.description}
        """
    else:
        doc = f"""\
        {self.option}: `{type_str}`
            {self.description} (default: `{self.default_value!r}`).
        """
    return textwrap.dedent(doc)


class RegisteredOption[T](NamedTuple):
    """A registered configuration option."""

    option: str
    default_value: T
    description: str
    validate: Callable[[T, SettingsManager], None]
    type: object

    describe = describe


def check_and_get_environ_var[T](
    key: str,
    default_value: str,
    allowed_values: Iterable[str] | None = None,
    cast: Callable[[Any], T] | type[Enum] = lambda x: x,
) -> T:
    """Get the environment variable and return it as a (potentially) non-string, usable value.

    Parameters
    ----------
    key
        The environment variable name.
    default_value
        The default value for `os.environ.get`.
    allowed_values
        Allowable string values, by default None.
    cast
        Casting from the string to a (potentially different) python object,
        by default lambda x: x.

    Returns
    -------
    The casted value.
    """
    environ_value_or_default_value = os.environ.get(key, default_value)
    if allowed_values is not None and environ_value_or_default_value not in allowed_values:
        msg = (
            f"Value {environ_value_or_default_value!r} is not in allowed {allowed_values} "
            f"for environment variable {key}. Default {default_value} will be used."
        )
        import warnings

        warnings.warn(msg, UserWarning)
        environ_value_or_default_value = default_value
    return (
        cast(environ_value_or_default_value)
        if not isinstance(cast, type(Enum))
        else cast[environ_value_or_default_value]
    )


def check_and_get_bool(option: str, default_value: bool) -> bool:
    """Get a boolean setting from environment variable."""
    return check_and_get_environ_var(
        f"ZDATA_{option.upper()}",
        str(int(default_value)),
        ["0", "1"],
        lambda x: bool(int(x)),
    )


def check_and_get_int(option: str, default_value: int) -> int:
    """Get an integer setting from environment variable."""
    return check_and_get_environ_var(
        f"ZDATA_{option.upper()}",
        str(int(default_value)),
        None,
        lambda x: int(x),
    )


_docstring = """
This manager allows users to customize settings for the zdata package.
Settings here will generally be for advanced use-cases and should be used with caution.

The following options are available:

{options_description}

For setting an option please use :func:`~zdata.settings.override` (local) or set the
above attributes directly (global) i.e., `zdata.settings.my_setting = foo`.
For assignment by environment variable, use the variable name in all caps with
`ZDATA_` as the prefix before import of :mod:`zdata`.
For boolean environment variable setting, use 1 for `True` and 0 for `False`.
"""


@dataclass
class SettingsManager:
    """Manager for zdata configuration settings."""

    _registered_options: dict[str, RegisteredOption] = field(default_factory=dict)
    _deprecated_options: dict[str, DeprecatedOption] = field(default_factory=dict)
    _config: dict[str, object] = field(default_factory=dict)
    __doc_tmpl__: str = _docstring

    def describe(
        self,
        option: str | Iterable[str] | None = None,
        *,
        should_print_description: bool = True,
        as_rst: bool = False,
    ) -> str:
        """Print and/or return a (string) description of the option(s).

        Parameters
        ----------
        option
            Option(s) to be described, by default None (i.e., do all options).
        should_print_description
            Whether or not to print the description in addition to returning it.
        as_rst
            Whether to format as reStructuredText.

        Returns
        -------
        The description.
        """
        describe_func = partial(
            self.describe,
            should_print_description=should_print_description,
            as_rst=as_rst,
        )
        if option is None:
            return describe_func(self._registered_options.keys())
        if isinstance(option, Iterable) and not isinstance(option, str):
            return "\n".join([describe_func(k) for k in option])
        registered_option = self._registered_options[option]
        doc = registered_option.describe(as_rst=as_rst).rstrip("\n")
        if option in self._deprecated_options:
            opt = self._deprecated_options[option]
            if opt.message is not None:
                doc += f" *{opt.message}"
            doc += f" {option} will be removed in {opt.removal_version}.*"
        if should_print_description:
            print(doc)
        return doc

    def deprecate(
        self, option: str, removal_version: str, message: str | None = None
    ) -> None:
        """Deprecate options with a message at a version.

        Parameters
        ----------
        option
            Which option should be deprecated.
        removal_version
            The version targeted for removal.
        message
            A custom message.
        """
        self._deprecated_options[option] = DeprecatedOption(
            option, message, removal_version
        )

    def register[T](
        self,
        option: str,
        *,
        default_value: T,
        description: str,
        validate: Callable[[T, Self], None],
        option_type: object | None = None,
        get_from_env: Callable[[str, T], T] = lambda x, y: y,
    ) -> None:
        """Register an option so it can be set/described etc. by end-users.

        Parameters
        ----------
        option
            Option to be set.
        default_value
            Default value with which to set the option.
        description
            Description to be used in the docstring.
        validate
            A function which raises a `ValueError` or `TypeError` if the value is invalid.
        option_type
            Optional override for the option type to be displayed.
            Otherwise `type(default_value)`.
        get_from_env
            An optional function which takes as arguments the name of the option and a
            default value and returns the value from the environment variable
            `ZDATA_CAPS_OPTION` (or default if not present).
            Default behavior is to return `default_value` without checking the environment.
        """
        try:
            validate(default_value, self)
        except (ValueError, TypeError) as e:
            e.add_note(f"for option {option!r}")
            raise e
        option_type = type(default_value) if option_type is None else option_type
        self._registered_options[option] = RegisteredOption(
            option, default_value, description, validate, option_type
        )
        self._config[option] = get_from_env(option, default_value)
        self._update_override_function_for_new_option(option)

    def _update_override_function_for_new_option(self, option: str) -> None:
        """Update the override function signature when a new option is registered."""
        option_type = self._registered_options[option].type
        # Update annotations for type checking.
        self.override.__annotations__[option] = option_type
        # __signature__ needs to be updated for tab autocompletion in IPython.
        self.override.__func__.__signature__ = signature(self.override).replace(
            parameters=[
                Parameter(name="self", kind=Parameter.POSITIONAL_ONLY),
                *[
                    Parameter(
                        name=k,
                        annotation=option_type,
                        kind=Parameter.KEYWORD_ONLY,
                    )
                    for k in self._registered_options
                ],
            ]
        )
        # Update docstring for `SettingsManager.override` as well.
        doc = cast("str", self.override.__doc__)
        insert_index = doc.find("\n        Yields")
        option_docstring = "\t" + "\t".join(
            self.describe(option, should_print_description=False).splitlines(
                keepends=True
            )
        )
        self.override.__func__.__doc__ = (
            f"{doc[:insert_index]}\n{option_docstring}{doc[insert_index:]}"
        )

    def __setattr__(self, option: str, val: object) -> None:
        """Set an option to a value.

        Parameters
        ----------
        option
            Option to be set.
        val
            Value with which to set the option.

        Raises
        ------
        AttributeError
            If the option has not been registered, this function will raise an error.
        """
        from dataclasses import fields

        if option in {f.name for f in fields(self)}:
            return super().__setattr__(option, val)
        elif option not in self._registered_options:
            msg = (
                f"{option} is not an available option for zdata. "
                "Please open an issue if you believe this is a mistake."
            )
            raise AttributeError(msg)
        registered_option = self._registered_options[option]
        registered_option.validate(val, self)
        self._config[option] = val

    def __getattr__(self, option: str) -> object:
        """Get the option's value.

        Parameters
        ----------
        option
            Option to be got.

        Returns
        -------
        Value of the option.
        """
        if option in self._deprecated_options:
            deprecated = self._deprecated_options[option]
            msg = (
                f"{option!r} will be removed in {deprecated.removal_version}. "
                f"{deprecated.message}"
            )
            import warnings

            warnings.warn(msg, FutureWarning)
        if option in self._config:
            return self._config[option]
        msg = f"{option} not found."
        raise AttributeError(msg)

    def __dir__(self) -> Iterable[str]:
        """Return list of available options."""
        from dataclasses import fields

        return sorted((*[f.name for f in fields(self)], *self._config.keys()))

    def reset(self, option: Iterable[str] | str) -> None:
        """Reset option(s) to its (their) default value(s).

        Parameters
        ----------
        option
            The option(s) to be reset.
        """
        if isinstance(option, Iterable) and not isinstance(option, str):
            for opt in option:
                self.reset(opt)
        else:
            self._config[option] = self._registered_options[option].default_value

    @contextmanager
    def override(self, **overrides: Any):
        """Provide local override via keyword arguments as a context manager.

        Yields
        ------
        None
        """
        restore = {a: getattr(self, a) for a in overrides}
        try:
            # Preserve order so that settings that depend on each other can be
            # overridden together
            for k in self._config:
                if k in overrides:
                    setattr(self, k, overrides.get(k))
            yield None
        finally:
            for attr, value in restore.items():
                setattr(self, attr, value)

    def __repr__(self) -> str:
        """Return string representation of settings."""
        params = "".join(f"\t{k}={v!r},\n" for k, v in self._config.items())
        return f"{type(self).__name__}(\n{params}\n)"

    @property
    def __doc__(self) -> str:
        """Generate docstring from registered options."""
        in_sphinx = any("/sphinx/" in frame.filename for frame in inspect.stack())
        options_description = self.describe(
            should_print_description=False, as_rst=in_sphinx
        )
        return self.__doc_tmpl__.format(options_description=options_description)


settings = SettingsManager()

##################################################################################
# PLACE REGISTERED SETTINGS HERE SO THEY CAN BE PICKED UP FOR DOCSTRING CREATION #
##################################################################################


def gen_validator[V](_type: type[V] | tuple[type[V], ...], /) -> Callable[[V, SettingsManager], None]:
    """Generate a validator function for a given type."""

    def validate_type(val: V, settings: SettingsManager) -> None:
        if not isinstance(val, _type):
            msg = f"{val} not valid {_type}"
            raise TypeError(msg)

    return validate_type


validate_bool = gen_validator(bool)
validate_int = gen_validator(int)


def validate_positive_int(val: int, settings: SettingsManager) -> None:
    """Validate that an integer is positive."""
    validate_int(val, settings)
    if val <= 0:
        msg = f"{val} must be positive"
        raise ValueError(msg)


def validate_max_rows_per_chunk(val: int, settings: SettingsManager) -> None:
    """Validate max_rows_per_chunk setting."""
    validate_positive_int(val, settings)
    # Reasonable upper bound to prevent memory issues
    if val > 100000:
        msg = f"max_rows_per_chunk ({val}) is very large and may cause memory issues"
        import warnings

        warnings.warn(msg, UserWarning)


def validate_block_rows(val: int, settings: SettingsManager) -> None:
    """Validate block_rows setting."""
    validate_positive_int(val, settings)
    # Block rows should be a power of 2 for optimal performance
    if val & (val - 1) != 0:
        msg = f"block_rows ({val}) is not a power of 2, which may impact performance"
        import warnings

        warnings.warn(msg, UserWarning)


# Register settings
settings.register(
    "max_rows_per_chunk",
    default_value=8192,
    description="Maximum number of rows per chunk file",
    validate=validate_max_rows_per_chunk,
    get_from_env=check_and_get_int,
)

settings.register(
    "block_rows",
    default_value=16,
    description="Number of rows per block in the compressed format",
    validate=validate_block_rows,
    get_from_env=check_and_get_int,
)

settings.register(
    "chunk_cache_size",
    default_value=10,
    description="Number of chunk files to cache in memory for faster access",
    validate=validate_positive_int,
    get_from_env=check_and_get_int,
)

settings.register(
    "warn_on_large_queries",
    default_value=True,
    description="Whether to warn when querying a large number of rows (>50000)",
    validate=validate_bool,
    get_from_env=check_and_get_bool,
)

settings.register(
    "large_query_threshold",
    default_value=50000,
    description="Threshold for warning on large queries (in number of rows)",
    validate=validate_positive_int,
    get_from_env=check_and_get_int,
)

settings.register(
    "override_memory_check",
    default_value=False,
    description="If True, allow queries to proceed even if they exceed 80% of available memory (issues warning instead of raising MemoryError).",
    validate=validate_bool,
    get_from_env=check_and_get_bool,
)

settings.register(
    "max_workers",
    default_value=None,
    description="Maximum number of worker threads for parallel file reads. None means auto-detect based on CPU count. Set to 1 to disable parallelization.",
    validate=lambda val, s: None if val is None else validate_positive_int(val, s),
    get_from_env=lambda opt, default: None if os.environ.get(f"ZDATA_{opt.upper()}") is None else check_and_get_int(opt, default or 4),
)

