// Same plugin set and versions as the group's app on the restart branch, so the two Android
// projects in this repository never disagree about the toolchain.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
