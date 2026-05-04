package com.fractalov.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.AsyncTaskExecutor;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

@Configuration
@EnableAsync
@EnableScheduling
public class AsyncConfig {

    @Bean(name = "renderJobExecutor")
    public AsyncTaskExecutor renderJobExecutor(JobsProperties props) {
        ThreadPoolTaskExecutor exec = new ThreadPoolTaskExecutor();
        // Default = 1 because each render itself uses parallelStream over CPU cores;
        // raising worker count just contends for the same threads. Bump only if you
        // intentionally want pipelining (e.g. one render running, another encoding PNG).
        exec.setCorePoolSize(props.workerPoolSize());
        exec.setMaxPoolSize(props.workerPoolSize());
        exec.setQueueCapacity(0);
        exec.setThreadNamePrefix("render-job-");
        exec.setWaitForTasksToCompleteOnShutdown(true);
        exec.setAwaitTerminationSeconds(30);
        exec.initialize();
        return exec;
    }
}
