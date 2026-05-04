package com.example.myapplication.ui

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.myapplication.ui.screen.ComparisonScreen
import com.example.myapplication.ui.screen.HomeScreen
import com.example.myapplication.ui.screen.MlScreen
import com.example.myapplication.ui.screen.VariationsScreen

/**
 * Shell that wraps every screen with a consistent top bar + animated route
 * transitions.
 *
 * Why one shell, not per-screen Scaffolds:
 *   * The top bar's title and back-button visibility derive from the current
 *     route — keeping that derivation in one place removes a class of
 *     "title forgot to update" bugs.
 *   * Navigation animations chain better when the NavHost owns every
 *     transition; a per-screen Scaffold would re-frame the bar on each
 *     hop and cause flicker.
 *
 * Animation: subtle horizontal slide + crossfade. ``200ms`` is long enough
 * to read as motion, short enough not to feel laggy on a tap.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppShell() {
    val nav = rememberNavController()
    val currentEntry by nav.currentBackStackEntryAsState()
    val currentRoute = currentEntry?.destination?.route
    val showBack = currentRoute != null && currentRoute != Routes.HOME

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Text(
                        text = titleFor(currentRoute),
                        style = MaterialTheme.typography.titleLarge,
                    )
                },
                navigationIcon = {
                    if (showBack) {
                        IconButton(onClick = { nav.popBackStack() }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    scrolledContainerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                ),
            )
        },
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            NavHost(
                navController = nav,
                startDestination = Routes.HOME,
                enterTransition = { slideInRight() },
                exitTransition = { fadeOut(animationSpec = tween(200)) },
                popEnterTransition = { fadeIn(animationSpec = tween(200)) },
                popExitTransition = { slideOutRight() },
            ) {
                composable(Routes.HOME) {
                    HomeScreen(
                        onOpenMl = { nav.navigate(Routes.ML) },
                        onOpenComparison = { nav.navigate(Routes.COMPARISON) },
                        onOpenVariations = { recipeJson ->
                            nav.navigate(Routes.variations(recipeJson))
                        },
                    )
                }
                composable(Routes.ML) {
                    MlScreen(
                        onOpenVariations = { recipeJson ->
                            nav.navigate(Routes.variations(recipeJson))
                        },
                    )
                }
                composable(Routes.COMPARISON) {
                    ComparisonScreen(
                        onOpenVariations = { recipeJson ->
                            nav.navigate(Routes.variations(recipeJson))
                        },
                    )
                }
                composable(
                    route = Routes.VARIATIONS_PATTERN,
                    arguments = listOf(
                        navArgument(Routes.VARIATIONS_ARG) {
                            type = NavType.StringType
                            nullable = true
                            defaultValue = null
                        },
                    ),
                ) { entry ->
                    val raw = entry.arguments?.getString(Routes.VARIATIONS_ARG)
                    val decoded = raw?.let { java.net.URLDecoder.decode(it, Charsets.UTF_8) }
                    VariationsScreen(seedRecipeJson = decoded)
                }
            }
        }
    }
}

private fun titleFor(route: String?): String = when (route?.substringBefore('?')) {
    Routes.HOME -> "fractalov"
    Routes.ML -> "ML from image"
    Routes.COMPARISON -> "Predicted vs reconstructed"
    "variations" -> "Variations"
    else -> "fractalov"
}

private fun AnimatedContentTransitionScope<NavBackStackEntry>.slideInRight() =
    slideIntoContainer(
        towards = AnimatedContentTransitionScope.SlideDirection.Start,
        animationSpec = tween(durationMillis = 240),
        // 12 dp horizontal travel — perceived motion without ribbon-y drift.
        initialOffset = { fullWidth -> (fullWidth * 0.12).toInt() },
    ) + fadeIn(animationSpec = tween(durationMillis = 200))

private fun AnimatedContentTransitionScope<NavBackStackEntry>.slideOutRight() =
    slideOutOfContainer(
        towards = AnimatedContentTransitionScope.SlideDirection.End,
        animationSpec = tween(durationMillis = 200),
        targetOffset = { fullWidth -> (fullWidth * 0.12).toInt() },
    ) + fadeOut(animationSpec = tween(durationMillis = 160))

@Suppress("UnusedReceiverParameter") // kept for symmetry with neighbouring ext fns
private val unused: Unit = Unit
