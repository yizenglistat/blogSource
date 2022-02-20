# Source for yizengli.com [![example workflow](https://github.com/yizenglistat/blogsource/actions/workflows/autodeploy.yml/badge.svg)](https://yizengli.com)

OpenMod is a plugin framework for .NET.

It supports authorization, plugin configurations, internalization, command handling and much more. OpenMod can be used for games, bot frameworks, web servers or anything else and has official implementations for Unturned, Rust (WIP) and a standalone console.

For a list of available plugins, visit [openmod-plugins](https://openmod.github.io/openmod-plugins).

## Features
OpenMod is based on modern C# code and best practices.
- Modern API for plugin development with C# and Unity best practices
- Plugin installation with [NuGet](https://nuget.org)
- Can self update with NuGet
- Based on [.NET Generic Host](https://docs.microsoft.com/en-us/aspnet/core/fundamentals/host/generic-host)
- IoC and Dependency Injection using Autofac
- Configure OpenMod and plugins with yaml configurations, environment variables, commandline arguments, etc.
- Serilog for logging, including rich configuration options via logging.yml

## Getting Started
To get started, visit the [OpenMod Documentation](https://openmod.github.io/openmod-docs/).

If you would like to install OpenMod, installation guides for the following platforms are available:
- [Unturned](https://openmod.github.io/openmod-docs/userdoc/installation/unturned.html)

If you want to make plugins for OpenMod, you can get started by reading the [Making your first plugin](https://openmod.github.io/openmod-docs/devdoc/guides/getting-started.html) page.

## Supported Games
Currently Unturned is the only supported game. More games might follow in the future. OpenMod can work side-by-side with RocketMod. 

## License
See [LICENSE](LICENSE) file for more information.

## Build Status [![LGTM alerts](https://img.shields.io/lgtm/alerts/github/openmod/openmod?style=for-the-badge)](https://lgtm.com/projects/g/openmod/openmod/)
### Framework
[![OpenMod.API](https://github.com/openmod/OpenMod/workflows/OpenMod.API/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.API) [![OpenMod.Bootstrapper](https://github.com/openmod/OpenMod/workflows/OpenMod.Bootstrapper/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Bootstrapper) [![OpenMod.NuGet](https://github.com/openmod/OpenMod/workflows/OpenMod.NuGet/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.NuGet) [![OpenMod.Core](https://github.com/openmod/OpenMod/workflows/OpenMod.Core/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Core) [![OpenMod.Runtime](https://github.com/openmod/OpenMod/workflows/OpenMod.Runtime/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Runtime) [![OpenMod.Analyzers](https://github.com/openmod/OpenMod/workflows/OpenMod.Analyzers/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Analyzers) [![OpenMod.Templates](https://github.com/openmod/OpenMod/workflows/OpenMod.Templates/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Templates) [![OpenMod.EntityFrameworkCore](https://github.com/openmod/OpenMod/workflows/OpenMod.EntityFrameworkCore/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.EntityFrameworkCore)

### Extensions
[![OpenMod.Extensions.Games.Abstractions](https://github.com/openmod/OpenMod/workflows/OpenMod.Extensions.Games.Abstractions/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Extensions.Games.Abstractions) [![OpenMod.Extensions.Economy.Abstractions](https://github.com/openmod/OpenMod/workflows/OpenMod.Extensions.Economy.Abstractions/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Extensions.Economy.Abstractions)

### Standalone
[![OpenMod.Standalone](https://github.com/openmod/OpenMod/workflows/OpenMod.Standalone/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Standalone)

### UnityEngine
[![OpenMod.UnityEngine.Redist](https://github.com/openmod/OpenMod/workflows/OpenMod.UnityEngine.Redist/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.UnityEngine.Redist) [![OpenMod.UniTask](https://github.com/openmod/OpenMod/workflows/OpenMod.UniTask/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.UniTask) [![OpenMod.UnityEngine](https://github.com/openmod/OpenMod/workflows/OpenMod.UnityEngine/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.UnityEngine)

### Unturned
[![OpenMod.Unturned.Redist](https://github.com/openmod/OpenMod/workflows/OpenMod.Unturned.Redist/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Unturned.Redist) [![OpenMod.Unturned](https://github.com/openmod/OpenMod/workflows/OpenMod.Unturned/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Unturned) [![OpenMod.Unturned.Module](https://github.com/openmod/OpenMod/workflows/OpenMod.Unturned.Module/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Unturned.Module)

### Rust
[![OpenMod.Rust.Redist](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust.Redist/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust.Redist) [![OpenMod.Rust](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust) [![OpenMod.Rust.Oxide.Redist](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust.Oxide.Redist/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust.Oxide.Redist) [![OpenMod.Rust.Oxide](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust.Oxide/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust.Oxide) [![OpenMod.Rust.Oxide.Extension](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust.Oxide.Extension/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust.Oxide.Extension) [![OpenMod.Rust.Oxide.PermissionLink](https://github.com/openmod/OpenMod/workflows/OpenMod.Rust.Oxide.PermissionLink/badge.svg)](https://github.com/openmod/OpenMod/actions?query=workflow%3AOpenMod.Rust.Oxide.PermissionLink)