# C++日志模块 loguru

loguru一共只需要两个源文件: loguru.hpp 和 loguru.cpp.
链接的时候还需要: -lpthread -ldl
支持断言: CHECK_F(fp != nullptr, "Failed to open '%s'", filename)
支持中断: ABORT_F("Something went wrong, debug value is %d", value). (没有成功测试这个宏)

## 使用教程


将`loguru::g_stderr_verbosity`设置成下面这些变量，开启相应的日志等级。

```c++
loguru::g_stderr_verbosity = 9;                            // print everything
loguru::g_stderr_verbosity = loguru::Verbosity_OFF;        // not print anthing
```

``` c# 
// You may use Verbosity_OFF on g_stderr_verbosity, but for nothing else!
Verbosity_OFF     = -9, // Never do LOG_F(OFF)

// Prefer to use ABORT_F or ABORT_S over LOG_F(FATAL) or LOG_S(FATAL).
Verbosity_FATAL   = -3,
Verbosity_ERROR   = -2,
Verbosity_WARNING = -1,

// Normal messages. By default written to stderr.
Verbosity_INFO    =  0,

// Same as Verbosity_INFO in every way.
Verbosity_0       =  0,

// Don not use higher verbosity levels, as that will make grepping log files harder.
Verbosity_MAX     = +9,
```

将不同等级的日志记录到不同的文件里
```c++
loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
```

标准输出,后缀_F类似于`printf`
```c++
LOG_F(INFO, "I'm hungry for some %.3f!", 3.14159);
```

判断程序，如果真，则输出
```c++
LOG_IF_F(ERROR, true, "Will only show if badness happens");
```

断言，会中断程序
```c++
CHECK_F(1 == 0, "Assertion 1 == 0 failed.\n '%s'\n '%d' ", a.c_str(), 1000);
```


## 示例代码
```c++
/// g++ demo.cpp loguru.cpp -lpthread -ldl

#include "loguru.hpp"
#include <cstdio>
#include <string>

int main(int argc,char *argv[]){
    loguru::g_stderr_verbosity = 9;

    // Optional, but useful to time-stamp the start of the log.
    // Will also detect verbosity level on command line as -v.
    // loguru::init(argc, argv);

    // Put every log message in "everything.log":
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);

    // Only log INFO, WARNING, ERROR and FATAL to "latest_readable.log":
    loguru::add_file("latest_readable.log", loguru::Truncate, loguru::Verbosity_INFO);

    // Only show most relevant things on stderr:
    // loguru::g_stderr_verbosity = 1;

    LOG_SCOPE_F(INFO, "Will indent all log messages within this scope.");
    LOG_F(INFO, "I'm hungry for some %.3f!", 3.14159);
    LOG_F(2, "Will only show if verbosity is 2 or higher");
    // VLOG_F(get_log_level(), "Use vlog for dynamic log level (integer in the range 0-9, inclusive)");
    LOG_IF_F(ERROR, true, "Will only show if badness happens");
    // auto fp = fopen(filename, "r");
    std::string a = "Here is a string."; 
    //CHECK_F(1 == 0, "Assertion 1 == 0 failed.\n '%s'\n '%d' ", a.c_str(), 1000);
    /// CHECK_GT_F(length, 0); // Will print the value of `length` on failure.
    /// CHECK_EQ_F(a, b, "You can also supply a custom message, like to print something: %d", a + b);

    // Each function also comes with a version prefixed with D for Debug:
    /// DCHECK_F(expensive_check(x)); // Only checked #if !NDEBUG
    DLOG_F(INFO, "Only written in debug-builds");

    // Turn off writing to stderr:
    loguru::g_stderr_verbosity = loguru::Verbosity_OFF;

    // Turn off writing err/warn in red:
    loguru::g_colorlogtostderr = false;
    return 0;
}
```


