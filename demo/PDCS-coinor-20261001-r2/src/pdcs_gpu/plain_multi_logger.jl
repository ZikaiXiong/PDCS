using Logging

struct PlainMultiLogger <: AbstractLogger
    io_list::Vector{IO}
    level::Logging.LogLevel
end

Logging.min_enabled_level(logger::PlainMultiLogger) = logger.level

function Logging.shouldlog(logger::PlainMultiLogger, level, _module, group, id)
    return level >= logger.level
end

function Logging.handle_message(
    logger::PlainMultiLogger,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    level < logger.level && return
    for io in logger.io_list
        println(io, message)
        for (key, value) in pairs(kwargs)
            print(io, "  ", key, " = ")
            show(io, value)
            println(io)
        end
        if io isa IOStream && io != stdout
            flush(io)
        end
    end
    return
end
