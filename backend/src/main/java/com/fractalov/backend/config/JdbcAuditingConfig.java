package com.fractalov.backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.jdbc.repository.config.EnableJdbcAuditing;

/**
 * Enables {@code @CreatedDate}/{@code @LastModifiedDate} population on entities.
 * Kept in a dedicated class so it can be excluded in slice tests if needed.
 */
@Configuration
@EnableJdbcAuditing
public class JdbcAuditingConfig {
}
