<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('model_feedbacks', function (Blueprint $table) {
            $table->id();
            $table->char('song_id', 10);
            $table->boolean('is_correct'); // True (Listen >30s) or False (Skip <30s / report)
            $table->string('feedback_type')->nullable(); // explicit, implicit, implicit_skip
            $table->timestamps();

            // Foreign key to songs table
            $table->foreign('song_id')->references('id')->on('songs')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('model_feedbacks');
    }
};
