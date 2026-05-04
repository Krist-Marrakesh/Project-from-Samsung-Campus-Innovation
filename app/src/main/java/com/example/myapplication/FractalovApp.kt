package com.example.myapplication

import android.app.Application
import com.example.myapplication.network.FractalovApi

/**
 * Manual DI root — the application is small enough that a single Application
 * subclass holding singletons beats wiring up Hilt or Koin. Anything a
 * ViewModel needs goes through here via `application as FractalovApp`.
 */
class FractalovApp : Application() {

    val api: FractalovApi by lazy {
        FractalovApi(baseUrl = BuildConfig.BACKEND_BASE_URL)
    }

    override fun onTerminate() {
        api.close()
        super.onTerminate()
    }
}
