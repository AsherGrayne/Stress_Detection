package com.stressdetection.iot.config;

import java.io.IOException;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Chrome may send {@code Access-Control-Request-Private-Network} on preflight when the web app
 * (e.g. Flutter on localhost) calls the API on another local port — respond so the browser allows it.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class PrivateNetworkCorsFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        if ("OPTIONS".equalsIgnoreCase(request.getMethod())
                && "true".equalsIgnoreCase(request.getHeader("Access-Control-Request-Private-Network"))) {
            response.addHeader("Access-Control-Allow-Private-Network", "true");
        }
        filterChain.doFilter(request, response);
    }
}
